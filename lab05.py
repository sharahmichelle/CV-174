import cv2
import numpy as np
import random
import time

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Game constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
GROUND_HEIGHT = 350
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 80
OBSTACLE_WIDTH = 40
MIN_OBSTACLE_GAP = 300
OBSTACLE_SPAWN_INTERVAL = 2.5

class Player:
    def __init__(self):
        self.x = 100
        self.y = GROUND_HEIGHT - PLAYER_HEIGHT
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.is_jumping = False
        self.jump_velocity = -20
        self.gravity = 1
        self.score = 0
        self.game_over = False
    
    def update(self, is_smiling):
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            
            # prevents sinking below the ground
            if self.y >= GROUND_HEIGHT - self.height:
                self.y = GROUND_HEIGHT - self.height
                self.is_jumping = False
        
        if is_smiling and not self.is_jumping:
            self.jump()
    
    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = -20
    
    def draw(self, frame):
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.width), int(self.y + self.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        eye_y = int(self.y + 20)
        cv2.circle(frame, (int(self.x + 15), eye_y), 5, (0, 0, 0), -1)
        cv2.circle(frame, (int(self.x + 35), eye_y), 5, (0, 0, 0), -1)
        mouth_y = int(self.y + 50)
        if self.is_jumping:
            cv2.ellipse(frame, (int(self.x + 25), mouth_y), (10, 5), 0, 0, 180, (0, 0, 0), 2)
        else:
            cv2.ellipse(frame, (int(self.x + 25), mouth_y), (10, 5), 0, 0, -180, (0, 0, 0), 2)

class Obstacle:
    def __init__(self):
        self.width = OBSTACLE_WIDTH
        self.height = random.choice([40, 50, 60])
        self.x = SCREEN_WIDTH
        self.y = GROUND_HEIGHT - self.height
        self.speed = 5
        self.passed = False
    
    # simulate obstacle moving across the screen toward the player
    def update(self):
        self.x -= self.speed
    
    def draw(self, frame):
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.width), int(self.y + self.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    # determines if the rectangle representing the player collides with the rectangle representing the obstacle
    def collides_with(self, player):
        if (player.x + 10 < self.x + self.width and  # right side of the player is to the left of the right side of the obstacle
            player.x + player.width - 10 > self.x and  # left side of the player is to the right of the left side of the obstacle
            player.y + 10 < self.y + self.height and  # top side of the player is above the bottom side of the obstacle
            player.y + player.height - 10 > self.y):  # bottom side of the player is below the top side of the obstacle
            return True
        return False

def detect_smile(face_roi, gray_face):
    mouths = mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)
    
    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]  # coordinates and dimensions of the detected mouth region
        # determines if the detected mouth is wide enough to indicate a smile
        face_height = face_roi[3]
        mouth_ratio = mh / face_height #how large the mouth is relative to the face height
        
        # greater than 20 = smile
        if mouth_ratio > 0.20:
            return True
    return False

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(1)
    
    # Create a named window that can be resized
    cv2.namedWindow('Smile to Jump', cv2.WINDOW_NORMAL) # Resizable window
    cv2.resizeWindow('Smile to Jump', SCREEN_WIDTH, SCREEN_HEIGHT) # starts with the desired dimensions
    
    player = Player()
    obstacles = []
    last_obstacle_time = time.time() # ensure that obstacles are spawned at the correct intervals
    
    # keep the game running
    while True:
        ret, cam_frame = cap.read()
        if not ret:
            break
        
        # Create a blank canvas with our desired dimensions
        game_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Flip and resize the camera frame to fit our game dimensions
        cam_frame = cv2.flip(cam_frame, 1)
        cam_frame = cv2.resize(cam_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Blend the camera feed with our game frame (50% opacity)
        game_frame = cv2.addWeighted(game_frame, 0.5, cam_frame, 0.5, 0)
        
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY) # convert to grayscale for face detection
        
        # Detect face and smile
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        is_smiling = False
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = (x, y, w, h)
            gray_face = gray[y:y+h, x:x+w]
            
            is_smiling = detect_smile(face_roi, gray_face)
            
            cv2.rectangle(game_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            expression_text = "Expression: Smile" if is_smiling else "Expression: Neutral"
            cv2.putText(game_frame, expression_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        player.update(is_smiling)
        
        current_time = time.time() # calculates time elapsed since the last obstacle was spawned
        # check if it's time to spawn a new obstacle
        if (current_time - last_obstacle_time > OBSTACLE_SPAWN_INTERVAL and 
            # ensures there is enough space between obstacles
            (not obstacles or (SCREEN_WIDTH - obstacles[-1].x) > MIN_OBSTACLE_GAP)):
            obstacles.append(Obstacle()) # spawn a new obstacle
            last_obstacle_time = current_time
            
            # increase the speed of the last obstacle if the player has scored greater than 0 and divsible by 5
            if player.score > 0 and player.score % 5 == 0:
                obstacles[-1].speed = min(7, obstacles[-1].speed * 1.1)
        
        # create copy of the obstacles list to avoid modifying it while iterating
        for obstacle in obstacles[:]:
            if not player.game_over:
                obstacle.update()
            
            obstacle.draw(game_frame)
            
            if obstacle.collides_with(player):
                player.game_over = True
            
            # removes obstacles that move off screen
            if obstacle.x + obstacle.width < 0:
                if not obstacle.passed:
                    player.score += 1
                    obstacle.passed = True
                obstacles.remove(obstacle)
        
        player.draw(game_frame) # render player as a green rectangle
        
        # Display score (top-left corner)
        cv2.putText(game_frame, f"Score: {player.score}", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Game over screen
        if player.game_over:
            game_over_text = "GAME OVER"
            score_text = f"Final Score: {player.score}"
            restart_text = "Press 'R' to restart"
            
            # Calculate text sizes
            (game_over_width, game_over_height), _ = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            (score_width, score_height), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            (restart_width, restart_height), _ = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Calculate positions
            center_x = SCREEN_WIDTH // 2
            start_y = SCREEN_HEIGHT // 2 - 50
            
            # Draw text
            cv2.putText(game_frame, game_over_text, 
                       (center_x - game_over_width // 2, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.putText(game_frame, score_text, 
                       (center_x - score_width // 2, start_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(game_frame, restart_text, 
                       (center_x - restart_width // 2, start_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instructions (bottom-left corner)
        cv2.putText(game_frame, "Smile to JUMP over obstacles", 
                   (30, SCREEN_HEIGHT - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show the game frame
        cv2.imshow('Smile to Jump', game_frame)
            
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('r') and player.game_over:
            player = Player()
            obstacles = []
            last_obstacle_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
