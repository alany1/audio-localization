import cv2
import numpy as np

if __name__ == "__main__":

    import cv2

    print("starting")
    # Open the first video
    video1 = cv2.VideoCapture('results/output_frames.mp4')

    # Open the second video
    video2 = cv2.VideoCapture('results/output.mp4')

    # Get the frame count and frame size of the first video
    frame_count1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size1 = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Get the frame count and frame size of the second video
    frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size2 = (int(video2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Set the output video size
    output_size = (224, 224)

    # Create the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/output_overlay.mp4', fourcc, 30, output_size)
    print("starting loop")
    # Loop through the frames of both videos
    for i in range(min(frame_count1, frame_count2)):
        # Read the frames
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # Resize the frames to a common size
        frame1_resized = cv2.resize(frame1, output_size)
        frame2_resized = cv2.resize(frame2, output_size)

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2HSV)

        # Define the range of "red" color
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        # Threshold the HSV image to get only "red" colors
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Bitwise AND the mask and original frame
        res = cv2.bitwise_and(frame2_resized, frame2_resized, mask=mask)
        result = cv2.addWeighted(frame1, 0.7, res, 0.3, 0)

        # Overlay the frames with increasing transparency for lower values of the second video
        # overlay = cv2.addWeighted(frame1_resized, .5, frame2_resized, 1 - .5, 0)

        # Write the overlayed frame to the output video
        out.write(result)
    print("done")
    # Release the videos and the output video writer
    video1.release()
    video2.release()
    out.release()
