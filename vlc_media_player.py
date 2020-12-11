import pafy
import vlc
import time

class VLCMediaPlayer:
    """VLCMedia Player
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


    def play(self):
        self._player.play()

    def is_playing(self):
        yield self._player.is_playing()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()

    def mute(self):
        self._player.audio_toggle_mute()



if __name__ == "__main__":
    # media_player = VLCMediaPlayer(mode="terminal")
    media_player = VLCMediaPlayer()
    media_player.play()

    ans = ""
    while ans != "q":
        ans = input("Action ?")
        if ans == "p":
            media_player.pause()
        elif ans == "s":
            media_player.stop()
        elif ans == "go":
            media_player.play()
        elif ans == "m":
            media_player.mute()


    # while media_player.is_playing():
    #     time.sleep(1)
