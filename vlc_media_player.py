import pafy
import vlc
import time


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

    def get_state(self):
        return self._player.get_state()

    def is_paused(self):
        return self.get_state() == vlc.State.Paused

    def is_stopped(self):
        return self.get_state() == vlc.State.Stopped

    def is_playing(self):
        return self._player.is_playing()

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


class VLCController:
    """VLC Controller
    Control a VLC Media Player
    """

    def __init__(self, url="https://www.youtube.com/watch?v=xtp4msMYi9s"):
        self.media_player = VLCMediaPlayer(url)
        self.last_seen = None

    def test_run(self):
        self.media_player.play()
        time.sleep(5)
        self.media_player.pause()
        time.sleep(3)

    def run(self, prediction):
        "Decide from prediction"

        if self.last_seen is None:
            decision = prediction
        else:
            decision = prediction
            self.last_seen = decision

    def action(self, decision):
        """Control VLC Media Player accordingly to decision"""
        actions = dict(
            l="rewind",
            peace="pause",
            y="play",
            F="mute",
            B="stop",
            up="fastforward"
        )


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
