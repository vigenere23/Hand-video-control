import pafy
import vlc


class VLCMediaPlayer:
    """VLCMedia Player
    Play a video with VLC, from a YouTube url
    """

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



if __name__ == "__main__":
    # media_player = VLCMediaPlayer(mode="terminal")
    # try:
    #     media_player.play()
    #     input("Quit ?")
    # except:
    #     print("Excepted")
    #     # media_player.stop()
    media_player = VLCMediaPlayer()
    media_player.play()
    input("Quit ?")
