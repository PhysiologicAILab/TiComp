import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
from pathlib import Path
home = str(Path.home())

base_dir = os.path.join(home, "dev/data/Demo/frames_for_demo_video")
outdir = os.path.join(base_dir, 'out')
if not os.path.exists(outdir):
    os.makedirs(outdir)

data_path_sota = "SOTA"
data_path_samcl = "SAM-CL"

data_path_raw_images = os.path.join(home, "dev/data/Demo/Processed/test/image")
raw_ext = '.npy'
img_ext = '.jpg'
fnames = os.listdir(os.path.join(base_dir, data_path_sota))
total_frame_count = len(fnames)
# video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (400, 1200))

'''
#ffmpeg -i %04d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p SAM-CL_Demo.mp4
ffmpeg -pattern_type glob -i '*.jpg' -c:v libx264 -framerate 30 -filter:v "setpts=PTS/4" SAM-CL_Demo.mp4
# ffmpeg -i SAM-CL_Demo.mp4 SAM-CL_Demo.gif
ffmpeg -i SAM-CL_Demo.mp4 -vf "fps=30,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 SAM-CL_Demo.gif
see -> https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
'''

for i in range(total_frame_count):
    fn = f'{i+1:04d}'
    img = np.load(os.path.join(data_path_raw_images, fn + raw_ext))
    img_sota = cv2.imread(os.path.join(base_dir, data_path_sota, fn + img_ext))
    img_samcl = cv2.imread(os.path.join(base_dir, data_path_samcl, fn + img_ext))
    img_sota = cv2.cvtColor(img_sota, cv2.COLOR_RGB2BGR)
    img_samcl = cv2.cvtColor(img_samcl, cv2.COLOR_RGB2BGR)
    img_sota = img_sota[58:426, 96:558]
    img_samcl = img_samcl[58:426, 96:558]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Thermal Matrix')
    ax[0].axis('off')

    ax[1].imshow(img_sota)
    ax[1].set_title('SOTA')
    ax[1].axis('off')

    ax[2].imshow(img_samcl)
    ax[2].set_title('SAM-CL (Ours)')
    ax[2].axis('off')

    plt.tight_layout()
    # # plt.show()
    plt.savefig(os.path.join(outdir, fn+'.jpg'))
    # break
    # put pixel buffer in numpy array
    # canvas = FigureCanvas(fig)
    # canvas.draw()
    # mat = np.array(canvas.renderer._renderer)
    # mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    # # write frame to video
    # video.write(mat)
    sys.stdout.write("Processing: " + str(i) + " of " + str(total_frame_count) + "\r")
    sys.stdout.flush()

    plt.close()
    plt.cla()


# # close video writer
# cv2.destroyAllWindows()
# video.release()
