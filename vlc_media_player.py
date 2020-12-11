import pafy
import vlc


class VLCMediaPlayer:
    """VLCMedia Player
    Play a video with VLC, from a YouTube url
    """

    def __init__(self, url="https://www.youtube.com/watch?v=xtp4msMYi9s"):
        self.url = url
        video = pafy.new(self.url).getbest()
        self.media_player = vlc.MediaPlayer(video.url)

    def play(self):
        self.media_player.play()


if __name__ == "__main__":
    media_player = VLCMediaPlayer()
    media_player.play()
    while True:
        pass
