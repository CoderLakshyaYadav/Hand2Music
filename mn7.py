import cv2
import numpy as np
import pygame
import mediapipe as mp
import multiprocessing as mp_proc

def draw_grid(img, rows, cols, lenti, color=(0, 0, 0), thickness=1):
    h, w = img.shape[:2]
    if lenti>30 or lenti==0:
        for j in range(0, w, w // cols):
            cv2.line(img, (j, 0), (j, h),  color, thickness)
    if lenti<100 and lenti>0:
        for j in range(0, int(2*h/3)+1, h//rows):
            cv2.line(img, (0,j),(w,j), color, thickness)

def harmonic(h, rows, yc):
    ygrid=[]
    for i in range(0, int(2*h/3)+1, h//rows):
        ygrid.append(i)
    if yc>0 and yc<ygrid[1]:
        return 5
    elif yc>ygrid[1] and yc<ygrid[2]:
        return 4
    elif yc>ygrid[2] and yc<ygrid[3]:
        return 3
    elif yc>ygrid[3] and yc<ygrid[4]:
        return 2
    else:
        return 1

def note_frequency(octave_range,frequency):
    i=0
    for octave in octave_range:
        for pitch in range(12):
            frequency[i]=round(440*2**((octave-4)+(pitch-9)/12),2)
            i+=1

# Function to generate sound based on finger coordinates
def music_generator(coord_queue,frequency):
    pygame.mixer.init()
    sample_rate, duration = 44100, 1.0
    xp = 0
    n=1
    plenti=0  # Changed from None to 0 to prevent TypeError during integer comparison
    sound = None  # Explicitly initialize sound
    while True:
        # Get the latest finger coordinates
        if not coord_queue.empty():
            # Drain the queue to process only the absolute latest frame (fixes latency buildup)
            while not coord_queue.empty():
                xi, yc, lenti, frame_width, frame_height = coord_queue.get()

            if lenti<30 and lenti>0:
                if not 100>plenti>30:
                    xi=xp            
                n=harmonic(frame_height, 6, yc)                    
            elif lenti>30 and lenti<100:
                xi=xp
                plenti=lenti
            else:
                n=1

            freq_to_play=frequency[min(xi//(frame_width//len(frequency)),len(frequency)-1)]  # Example frequency based on x coordinate

            # Generate sine wave for the current frequency
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 32767 * np.sin(2 * np.pi * n * freq_to_play * t)

            
            # Convert 1D wave to 2D (stereo)        
            stereo_wave = np.column_stack((wave, wave)).astype(np.int16)
    
            #stop prev sound
            if sound is not None:
                sound.stop()

            # Create sound object and play
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
            
            xp=xi
        #Adjust the sleep to control the frequency change rate
        # pygame.time.delay(200)  # Play for a short time

# Function to capture camera input and process finger coordinates
def camera_input(coord_queue,frequency):    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        lenti=0
        xi = yi = xt = yt = xc = yc = 0
        
        # Convert the frame to RGB for mediapipe hand tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get middle finger tip coordinates (landmark 8)
                xi,yi = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                xt,yt=int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])
                xc, yc = (xi+xt)/2, (yi+yt)/2
                lenti = np.hypot(np.abs(xi-xt), np.abs(yi-yt))
                frame_width, frame_height=frame.shape[1],frame.shape[0]

                if lenti<100 and lenti>0:
                    cv2.line(frame, (xi, yi), (xt, yt), (0,0,255), 3)
                    cv2.circle(frame, (int(xc), int(yc)), 10, (0, 255, 255), -1)
                if lenti>30:
                    cv2.circle(frame, (int(xi), int(yi)), 10, (0,255,0), -1)
                    cv2.circle(frame, (int(xt), int(yt)), 10, (0,255,0), -1)
                elif lenti<30 and lenti>0:
                    cv2.circle(frame, (int(xc), int(yc)), 10, (255, 255, 0), -1)

                # Send the coordinates to the music generator process
                coord_queue.put((xi, yc, lenti, frame_width, frame_height))

                # Draw hand landmarks for visualization
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the camera feed with hand landmarks
        fframe=cv2.flip(frame,1)
        draw_grid(fframe, 6 ,len(frequency[:])-1, lenti)
        cv2.imshow('Hand Tracking', fframe)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to start both subprocesses
if __name__ == '__main__':
    octave_range=range(3,5)

    coord_queue = mp_proc.Queue()  # Shared queue for communication between processes
    frequency=mp_proc.Array('d',len(octave_range)*12)

    note_frequency(octave_range,frequency)
    # Create the two subprocesses
    p_music = mp_proc.Process(target=music_generator, args=(coord_queue,frequency))
    p_camera = mp_proc.Process(target=camera_input, args=(coord_queue,frequency))

    # Start the subprocesses
    p_music.start()
    p_camera.start()

    # Wait for both subprocesses to finish (this will keep the main program running)
    p_camera.join()
    p_music.terminate()  # Terminate the music generator when camera is done
