import imageio
from pathlib import Path

def make_demo():
    paths = [
        "explanations/gradcam_real.png",
        "explanations/lime_real.png",
        "explanations/gradcam_fake.png",
        "explanations/lime_fake.png"
    ]
    imgs = [imageio.v2.imread(p) for p in paths if Path(p).exists()]
    imageio.mimsave("assets/demo.gif", imgs, fps=1)  # fps=1: 1 second per frame

if __name__ == "__main__":
    make_demo()
