import pafy
import vlc
import time
from enum import IntEnum, auto

class Action(IntEnum):
    PLAY = 1
    PAUSE = 2
    STOP = 3
    MUTE = 4
    VOL_UP = 5
    VOL_DN = 6

class VLCMediaPlayer:
    """VLC Media Player
    Play a video with VLC, from a YouTube url
    """

    # Play
    # Stop
    # Pause
    # Mute
    # FF
    # Rewind

    def __init__(self, url="https://www.youtube.com/watch?v=xtp4msMYi9s", mode=None):
        self.url = url
        video = pafy.new(self.url).getbest()
        if str(mode).lower() == "terminal":
            self._player = vlc.MediaPlayer(video.url)
        else:
            instance = vlc.Instance("--loop")
            self._player = instance.media_player_new()
            media = instance.media_new(video.url)
            media.get_mrl()
            self._player.set_media(media)

    # STATES
    def get_state(self):
        return self._player.get_state()

    def is_paused(self):
        return self.get_state() == vlc.State.Paused

    def is_stopped(self):
        return self.get_state() == vlc.State.Stopped

    def is_playing(self):
        return self._player.is_playing()

    # ACTIONS (PLAYBACK)
    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()

    def mute(self):
        self._player.audio_toggle_mute()

    def next_frame(self):
        self._player.next_frame()

    # ACTIONS (VOLUME)
    def get_volume(self):
        return self._player.audio_get_volume()

    def set_volume(self, val):
        if val < 0:
            print("Volume minimal atteint")
        elif val > 100:
            print("Volume maximal atteint")
        else:
            self._player.audio_set_volume(val)

    def volume_up(self, offset=5):
        self.set_volume(self.get_volume() + offset)

    def volume_dn(self, offset=5):
        self.set_volume(self.get_volume() - offset)


class VLCController:
    """VLC Controller
    Control a VLC Media Player
    """

    def __init__(self, url="https://www.youtube.com/watch?v=xtp4msMYi9s"):
        self.media_player = VLCMediaPlayer(url)
        self.last_seen = None
        self.actions_dict = dict(
            Y=Action.PLAY,
            peace=Action.PAUSE,
            opened=Action.STOP,
            F=Action.MUTE,
            L=Action.VOL_UP,
            pointing_up=Action.VOL_DN
        )
        self.media_player.play()

    def test_run(self):
        self.media_player.play()
        time.sleep(5)
        self.media_player.pause()
        time.sleep(3)

    def run(self, prediction):
        "Decide from prediction"

        # Not taking None in acount
        if prediction is None or prediction not in self.actions_dict.keys():
            return

        # Create decision and keep last_seen in decision variable
        decision = prediction

        self.last_seen = prediction

        # Act accordingly
        self.action(decision)

    def is_playing(self):
        return self.media_player.is_playing()

    def action(self, decision):
        """Control VLC Media Player accordingly to decision"""
        dec_act = self.actions_dict.get(decision, None)

        action_options = {
            Action.PLAY:self.media_player.play,
            Action.PAUSE:self.media_player.pause,
            Action.STOP:self.media_player.stop,
            Action.MUTE:self.media_player.mute,
            Action.VOL_UP:self.media_player.volume_up,
            Action.VOL_DN:self.media_player.volume_dn,
        }
        action_options[dec_act]()


if __name__ == "__main__":
    # media_player = VLCMediaPlayer(mode="terminal")
    # media_player = VLCMediaPlayer()
    # media_player.play()

    # ans = ""
    # while ans != "q":
    #     ans = input("Action ?")
    #     if ans == "p":
    #         media_player.pause()
    #     elif ans == "s":
    #         media_player.stop()
    #     elif ans == "go":
    #         media_player.play()
    #     elif ans == "m":
    #         media_player.mute()

    # while media_player.is_playing():
    #     time.sleep(1)

    control = VLCController()
    while True:
        control.test_run()
