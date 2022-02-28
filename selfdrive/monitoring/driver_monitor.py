from math import atan2

from cereal import car
# from common.numpy_fast import interp
from common.realtime import DT_DMON
from selfdrive.hardware import TICI
from common.filter_simple import FirstOrderFilter
from common.stat_live import RunningStatFilter

EventName = car.CarEvent.EventName

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS():
  def __init__(self, TICI=TICI, DT_DMON=DT_DMON):
    self._DT_DMON = DT_DMON
    self._AWARENESS_TIME = 35. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 12.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

    self._FACE_THRESHOLD = 0.5
    self._PARTIAL_FACE_THRESHOLD = 0.765 if TICI else 0.43

    self._EE_THRESH11 = 0.75 if TICI else 0.4
    self._EE_THRESH12 = 3.25 if TICI else 2.45
    self._EE_THRESH21 = 0.01
    self._EE_THRESH22 = 0.35

    # 0.46, 0.35
    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts


# model output refers to center of cropped image, so need to apply the x displacement offset
RESIZED_FOCAL = 320.0
H, W, FULL_W = 320, 160, 426

class DistractedType:
  NOT_DISTRACTED = 0
  BAD_POSE = 1
  BAD_BLINK = 2
  DISTRACTED_E2E = 3

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib, is_rhd):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0] + .5)*W - W + FULL_W, (pos_desc[1]+.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - FULL_W//2, RESIZED_FOCAL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, RESIZED_FOCAL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2] * (1 - 2 * int(is_rhd))  # lhd -> -=, rhd -> +=
  return roll_net, pitch, yaw

class DriverStatus():
  def __init__(self, rhd=False, settings=DRIVER_MONITOR_SETTINGS()):
    # init policy settings
    self.settings = settings

    # init driver status
    self.ee1_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee2_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee1_calibrated = False
    self.ee2_calibrated = False

    self.eev1 = 0.
    self.eev2 = 1.

    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.face_detected = False
    self.face_partial = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.is_model_uncertain = False
    self.low_std = True
    self.hi_stds = 0
    self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME

    self._set_timers(active_monitoring=True)

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = self.settings._AWARENESS_PRE_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.threshold_prompt = self.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.step_change = self.settings._DT_DMON / self.settings._AWARENESS_TIME
      self.active_monitoring_mode = False

  def _is_driver_distracted(self):
    if self.ee1_calibrated:
      ee1_dist = self.eev1 > self.ee1_offseter.filtered_stat.M * self.settings._EE_THRESH12
    else:
      ee1_dist = self.eev1 > self.settings._EE_THRESH11

    if self.ee2_calibrated:
      ee2_dist = self.eev2 < self.ee2_offseter.filtered_stat.M * self.settings._EE_THRESH22
    else:
      ee2_dist = self.eev2 < self.settings._EE_THRESH21

    if ee1_dist or ee2_dist:
      return DistractedType.DISTRACTED_E2E
    else:
      return DistractedType.NOT_DISTRACTED

  # def set_policy(self, model_data, car_speed):
  #   pass

  def get_pose(self, driver_state, cal_rpy, car_speed, op_engaged):
    if not all(len(x) > 0 for x in (driver_state.faceOrientation, driver_state.facePosition,
                                    driver_state.faceOrientationStd, driver_state.facePositionStd)):
      return

    self.face_partial = driver_state.partialFace > self.settings._PARTIAL_FACE_THRESHOLD
    self.face_detected = driver_state.faceProb > self.settings._FACE_THRESHOLD or self.face_partial

    self.eev1 = driver_state.notReadyProb[1]
    self.eev2 = driver_state.readyProb[0]

    self.low_std = not self.face_partial

    self.driver_distracted = self._is_driver_distracted() > 0 and \
                                   driver_state.faceProb > self.settings._FACE_THRESHOLD and self.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.low_std and (not op_engaged or not self.driver_distracted):
      self.ee1_offseter.push_and_update(self.eev1)
      self.ee2_offseter.push_and_update(self.eev2)

    self.ee1_calibrated = self.ee1_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee2_calibrated = self.ee2_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not self.is_model_uncertain)
    if self.face_detected and not self.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.low_std:
      self.hi_stds = 0

  def update(self, events, driver_engaged, ctrl_active, standstill):
    if (driver_engaged and ctrl_active):
      # reset only when on disengagement if red reached
      self.awareness = 1.
      self.awareness_active = 1.
      self.awareness_passive = 1.
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.low_std and self.awareness > 0):
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*(1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    standstill_exemption = (standstill or not ctrl_active) and self.awareness - self.step_change <= self.threshold_prompt
    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected
    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill and reaching orange
      if not standstill_exemption:
        self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = EventName.driverDistracted if self.active_monitoring_mode else EventName.driverUnresponsive
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = EventName.promptDriverDistracted if self.active_monitoring_mode else EventName.promptDriverUnresponsive
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      events.add(alert)
