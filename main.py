
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.metrics import dp
from functools import partial
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

# Set window size
Window.size = (400, 600)

# Configuration
class Config:
    SAVE_DIR = "captured_faces"
    UNKNOWN_DIR = os.path.join(SAVE_DIR, "unknown")
    KNOWN_DIR = os.path.join(SAVE_DIR, "known")
    CREDENTIALS_FILE = "credentials.pkl"
    CONFIDENCE_THRESHOLD = 0.7
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.SAVE_DIR, cls.UNKNOWN_DIR, cls.KNOWN_DIR]:
            os.makedirs(directory, exist_ok=True)

# Create directories
Config.setup_directories()

def show_popup(title, content):
    """Show a popup message"""
    popup = Popup(title=title,
                 content=Label(text=content),
                 size_hint=(None, None),
                 size=(300, 200))
    popup.open()

class FaceDetector:
    def __init__(self):
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, frame):
        """Detect faces in frame and return face locations"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def compare_faces(self, face1_path, face2):
        """Compare two faces using histogram comparison"""
        # Load the stored face
        face1 = cv2.imread(face1_path, cv2.IMREAD_GRAYSCALE)
        if face1 is None:
            return 0.0
            
        # Convert current face to grayscale
        face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        
        # Resize faces to same size
        face1 = cv2.resize(face1, (100, 100))
        face2_gray = cv2.resize(face2_gray, (100, 100))
        
        # Calculate histograms
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2_gray], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, similarity)  # Ensure non-negative value

COLORS = {
    'bg_dark': (0.12, 0.12, 0.12, 1),  # Dark background
    'card_dark': (0.16, 0.16, 0.16, 1),  # Slightly lighter dark for cards
    'primary': (0.33, 0.63, 1, 1),  # Bright blue
    'text_primary': (0.95, 0.95, 0.95, 1),  # Almost white
    'text_secondary': (0.7, 0.7, 0.7, 1),  # Light gray
    'input_bg': (0.2, 0.2, 0.2, 1),  # Dark input background
    'error': (0.93, 0.33, 0.33, 1)  # Red for errors
}

class StylizedTextInput(TextInput):
    def __init__(self, **kwargs):
        # Initialize TextInput first
        super().__init__(**kwargs)
        
        # Set properties after super initialization
        self.background_color = (1, 1, 1, 1)  # White background instead of transparent
        self.cursor_color = (0.2, 0.6, 1, 1)
        self.foreground_color = (0.1, 0.1, 0.1, 1)
        self.hint_text_color = (0.5, 0.5, 0.5, 1)
        self.padding = [20, 15]
        self.font_size = '16sp'
        
        # Initialize canvas
        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # White background
            self.background_rect = Rectangle(pos=self.pos, size=self.size)
            
        with self.canvas.after:
            Color(0.9, 0.9, 0.9, 1)
            self.border_line = Line(points=[0, 0, 0, 0], width=1.5)
        
        # Bind updates
        self.bind(pos=self.update_canvas)
        self.bind(size=self.update_canvas)
            
    def update_canvas(self, *args):
        # Update both background rectangle and border line
        self.background_rect.pos = self.pos
        self.background_rect.size = self.size
        
        # Update the border line position
        self.border_line.points = [
            self.x, self.y, 
            self.x + self.width, self.y
        ]

class StylizedButton(Button):
    def __init__(self, primary=True, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.font_size = '16sp'
        self.height = dp(50)
        self.size_hint_y = None
        self.primary = primary
        
        with self.canvas.before:
            self.border_color = Color(*COLORS['primary'])
            self.border_rectangle = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[5]
            )
            
            # Fill color based on primary/secondary style
            if primary:
                Color(*COLORS['primary'])
            else:
                Color(*COLORS['card_dark'])
            
            self.background_rectangle = RoundedRectangle(
                pos=(self.pos[0] + 1.5, self.pos[1] + 1.5),
                size=(self.size[0] - 3, self.size[1] - 3),
                radius=[5]
            )
            
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        
    def update_canvas(self, *args):
        self.border_rectangle.pos = self.pos
        self.border_rectangle.size = self.size
        self.background_rectangle.pos = (self.pos[0] + 1.5, self.pos[1] + 1.5)
        self.background_rectangle.size = (self.size[0] - 3, self.size[1] - 3)

    def on_press(self):
        anim = Animation(background_color=(0.8, 0.8, 0.8, 0.2), duration=0.1)
        anim.start(self)

    def on_release(self):
        anim = Animation(background_color=(0, 0, 0, 0), duration=0.1)
        anim.start(self)

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.clearcolor = COLORS['bg_dark']  # Set dark background
        
        # Main container with card effect
        self.card = BoxLayout(
            orientation='vertical',
            padding=[dp(30), dp(20)],
            spacing=dp(20),
            size_hint=(None, None),
            size=(dp(400), dp(500)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        
        # Add card background
        with self.card.canvas.before:
            Color(*COLORS['card_dark'])
            self.card_bg = RoundedRectangle(
                pos=self.card.pos,
                size=self.card.size,
                radius=[15]
            )
        self.card.bind(pos=self._update_card_bg, size=self._update_card_bg)
        
        # Title with custom font
        self.title = Label(
            text='Face Recognition',
            font_size='28sp',
            color=COLORS['text_primary'],
            size_hint_y=None,
            height=dp(50),
            bold=True
        )
        self.card.add_widget(self.title)
        
        # Subtitle
        self.subtitle = Label(
            text='Welcome back! Please login to your account',
            font_size='14sp',
            color=COLORS['text_secondary'],
            size_hint_y=None,
            height=dp(30)
        )
        self.card.add_widget(self.subtitle)
        
        # Form container
        self.form_layout = BoxLayout(
            orientation='vertical',
            spacing=dp(20),
            size_hint_y=None,
            height=dp(200),
            padding=[0, dp(20)]
        )
        
        # Username input - simplified version
        self.username_input = TextInput(
            multiline=False,
            hint_text='Username',
            size_hint_y=None,
            height=dp(40),
            background_normal='',
            background_active='',
            background_color=(1, 1, 1, 1),  # White background
            foreground_color=(0, 0, 0, 1),  # Black text
            hint_text_color=(0.5, 0.5, 0.5, 1),  # Gray hint text
            padding=[10, 10],  # Add some padding
            font_size='16sp'
        )
        self.form_layout.add_widget(self.username_input)
        
        # Password input - simplified version
        self.password_input = TextInput(
            multiline=False,
            password=True,
            hint_text='Password',
            size_hint_y=None,
            height=dp(40),
            background_normal='',
            background_active='',
            background_color=(1, 1, 1, 1),  # White background
            foreground_color=(0, 0, 0, 1),  # Black text
            hint_text_color=(0.5, 0.5, 0.5, 1),  # Gray hint text
            padding=[10, 10],  # Add some padding
            font_size='16sp'
        )
        self.form_layout.add_widget(self.password_input)
        
        # Forgot Password link
        self.forgot_password_btn = Button(
            text='Forgot Password?',
            size_hint_y=None,
            height=dp(30),
            background_color=(0, 0, 0, 0),
            color=COLORS['primary'],
            font_size='14sp',
            pos_hint={'right': 1}
        )
        self.forgot_password_btn.bind(on_press=self.forgot_password)
        self.form_layout.add_widget(self.forgot_password_btn)
        
        self.card.add_widget(self.form_layout)
        
        # Buttons container
        self.buttons_layout = BoxLayout(
            orientation='vertical',
            spacing=dp(15),
            size_hint_y=None,
            height=dp(120)
        )
        
        # Login button
        self.login_btn = StylizedButton(
            text='Login',
            primary=True,
            color=COLORS['text_primary']
        )
        self.login_btn.bind(on_press=self.login)
        self.buttons_layout.add_widget(self.login_btn)
        
        # Register button
        self.register_btn = StylizedButton(
            text='Register',
            primary=False,
            color=COLORS['primary']
        )
        self.register_btn.bind(on_press=self.goto_register)
        self.buttons_layout.add_widget(self.register_btn)
        
        self.card.add_widget(self.buttons_layout)
        
        # Loading indicator
        self.loading_label = Label(
            text='Logging in...',
            opacity=0,
            color=COLORS['text_secondary'],
            size_hint_y=None,
            height=dp(30)
        )
        self.card.add_widget(self.loading_label)
        
        # Add the card to the screen
        self.add_widget(self.card)
        
        # Set initial opacity for fade-in
        self.card.opacity = 0
        
        # Schedule the fade-in animation
        Clock.schedule_once(self._fade_in, 0.1)
    
    def _update_card_bg(self, instance, value):
        self.card_bg.pos = instance.pos
        self.card_bg.size = instance.size
        
    def _fade_in(self, dt):
        anim = Animation(opacity=1, duration=0.3)
        anim.start(self.card)
            
    def show_loading(self, show=True):
        animation = Animation(opacity=1 if show else 0, duration=0.2)
        animation.start(self.loading_label)
        
    def login(self, instance):
        username = self.username_input.text
        password = self.password_input.text
        
        # Show loading animation
        self.show_loading()
        
        # Simulate loading (replace with actual login logic)
        Clock.schedule_once(partial(self._handle_login, username, password), 1)
        
    def _handle_login(self, username, password, *args):
        self.show_loading(False)
        
        try:
            with open(Config.CREDENTIALS_FILE, 'rb') as f:
                credentials = pickle.load(f)
        except:
            credentials = {}
        
        if username in credentials:
            if isinstance(credentials[username], dict):
                stored_password = credentials[username]['password']
            else:
                stored_password = credentials[username]
                
            if password == stored_password:
                verify_screen = self.manager.get_screen('verify')
                verify_screen.username = username
                self.manager.transition.direction = 'left'
                self.manager.current = 'verify'
            else:
                self._show_error('Invalid password')
        else:
            self._show_error('Username not found')
            
    def _show_error(self, message):
        # Add subtle shake animation for error
        anim = Animation(x=self.x + 10, duration=0.1) + \
               Animation(x=self.x - 10, duration=0.1) + \
               Animation(x=self.x, duration=0.1)
        anim.start(self)
        show_popup('Error', message)
    
    def forgot_password(self, instance):
        self.manager.transition.direction = 'left'
        self.manager.current = 'forgot_password'
    
    def goto_register(self, instance):
        self.manager.transition.direction = 'left'
        self.manager.current = 'register'

class RegisterScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create the main layout first
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Add background color to the layout instead of the screen
        with self.layout.canvas.before:
            Color(*COLORS['bg_dark'])
            self.bg_rect = Rectangle(pos=self.layout.pos, size=self.layout.size)
            self.layout.bind(pos=self._update_rect, size=self._update_rect)
        
        self.face_detector = FaceDetector()
        
        # Title with updated styling
        self.layout.add_widget(Label(
            text='Register New User',
            font_size='24sp',
            size_hint_y=None,
            height='50dp',
            color=COLORS['text_primary']
        ))
        
        # Input fields
        self.inputs = {}
        for field in ['username', 'password', 'email']:
            self.inputs[field] = TextInput(
                multiline=False,
                hint_text=field.capitalize(),
                password=(field == 'password'),
                size_hint_y=None,
                height='40dp'
            )
            self.layout.add_widget(self.inputs[field])
        
        # Camera preview with border
        self.camera_container = BoxLayout(
            size_hint_y=0.6,
            padding=2
        )
        
        # Add background to camera container
        with self.camera_container.canvas.before:
            Color(*COLORS['card_dark'])
            self.camera_bg = Rectangle(
                pos=self.camera_container.pos, 
                size=self.camera_container.size
            )
            self.camera_container.bind(pos=self._update_camera_rect, 
                                     size=self._update_camera_rect)
        
        self.image = Image(size_hint_y=1)
        self.camera_container.add_widget(self.image)
        self.layout.add_widget(self.camera_container)
        
        # Status label with updated styling
        self.status_label = Label(
            text='Looking for face...',
            size_hint_y=None,
            height='30dp',
            color=COLORS['text_secondary']
        )
        self.layout.add_widget(self.status_label)
        
        # Styled buttons
        capture_btn = StylizedButton(
            text='Capture & Register',
            primary=True
        )
        capture_btn.bind(on_press=self.capture_and_register)
        self.layout.add_widget(capture_btn)
        
        back_btn = StylizedButton(
            text='Back to Login',
            primary=False
        )
        back_btn.bind(on_press=self.back_to_login)
        self.layout.add_widget(back_btn)
        
        # Add the layout to the screen
        self.add_widget(self.layout)
        
        self.capture = None
        self.capture_event = None
    
    def _update_rect(self, instance, value):
        """Update the background rectangle"""
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size

    def validate_email(self,email):
        """Vlidate for email"""
        return email.lower().endswith('@gmail.com')

    
    def _update_camera_rect(self, instance, value):
        """Update the camera container background"""
        self.camera_bg.pos = instance.pos
        self.camera_bg.size = instance.size
            
    def on_enter(self):
        """Start camera when entering screen"""
        self.start_camera()
        
    def on_leave(self):
        """Stop camera when leaving screen"""
        self.stop_camera()
        
    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture_event = Clock.schedule_interval(self.update_camera, 1.0/30.0)
        
    def stop_camera(self):
        if self.capture_event:
            self.capture_event.cancel()
        if self.capture:
            self.capture.release()
            
    def update_camera(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.status_label.text = 'Face detected'
                self.status_label.color = COLORS['primary']
                
            if len(faces) == 0:
                self.status_label.text = 'No face detected'
                self.status_label.color = COLORS['text_secondary']
            
            # Convert to texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            
    def capture_and_register(self, instance):
        values = {k: v.text.strip() for k, v in self.inputs.items()}
        
        if not all(values.values()):
            show_popup('Error', 'All fields are required')
            return

        if not self.validate_email(values['email']):
            show_popup('Error','Invalid Email.')
            return
            
        ret, frame = self.capture.read()
        if ret:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) == 0:
                show_popup('Error', 'No face detected')
                return
            
            if len(faces) > 1:
                show_popup('Error', 'Multiple faces detected')
                return
                
            try:
                with open(Config.CREDENTIALS_FILE, 'rb') as f:
                    credentials = pickle.load(f)
            except:
                credentials = {}
                
            if values['username'] in credentials:
                show_popup('Error', 'Username already exists')
                return
                
            # Save face image
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            image_path = os.path.join(Config.KNOWN_DIR, f"{values['username']}.jpg")
            cv2.imwrite(image_path, face_img)
            
            # Save credentials
            credentials[values['username']] = {
                'password': values['password'],
                'email': values['email']
            }
            
            with open(Config.CREDENTIALS_FILE, 'wb') as f:
                pickle.dump(credentials, f)
                
            show_popup('Success', 'Registration successful!')
            self.back_to_login(None)
            
    def back_to_login(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = 'login'

class VerifyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.username = None
        self.face_detector = FaceDetector()
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        self.layout.add_widget(Label(
            text='Face Verification',
            font_size='24sp',
            size_hint_y=None,
            height='50dp'
        ))
        
        # Camera preview
        self.image = Image(size_hint_y=0.7)
        self.layout.add_widget(self.image)
        
        # Status label
        self.status_label = Label(
            text='Looking for face...',
            size_hint_y=None,
            height='50dp'
        )
        self.layout.add_widget(self.status_label)
        
        self.add_widget(self.layout)
        
        self.capture = None
        self.capture_event = None
        self.unknown_face_saved = False  # Track if unknown face was saved
        
    def on_enter(self):
        self.start_verification()
        
    def on_leave(self):
        self.stop_verification()
        
    def start_verification(self):
        self.capture = cv2.VideoCapture(0)
        self.capture_event = Clock.schedule_interval(self.update_verification, 1.0/30.0)
        self.unknown_face_saved = False  # Reset flag when starting new verification
        
    def stop_verification(self):
        if self.capture_event:
            self.capture_event.cancel()
        if self.capture:
            self.capture.release()
            
    def save_unknown_face(self, frame, face_location, username):
        """Save unrecognized face to the unknown directory"""
        try:
            x, y, w, h = face_location
            face_img = frame[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"unknown_{username}_{timestamp}.jpg"
            image_path = os.path.join(Config.UNKNOWN_DIR, image_filename)
            cv2.imwrite(image_path, face_img)
            print(f"Saved unknown face to: {image_path}")
            return image_path
        except Exception as e:
            print(f"Error saving unknown face: {e}")
            return None
            
    def update_verification(self, dt):
        ret, frame = self.capture.read()
        if ret:
            faces = self.face_detector.detect_faces(frame)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Compare with stored face
                stored_image_path = os.path.join(Config.KNOWN_DIR, f"{self.username}.jpg")
                similarity = self.face_detector.compare_faces(stored_image_path, face_img)
                
                # Draw rectangle
                color = (0, 255, 0) if similarity >= Config.CONFIDENCE_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                self.status_label.text = f'Confidence: {similarity:.2f}'
                
                if similarity >= Config.CONFIDENCE_THRESHOLD:
                    self.verification_success()
                    return
                elif not self.unknown_face_saved and similarity < Config.CONFIDENCE_THRESHOLD:
                    # Save unknown face if confidence is low and we haven't saved it yet
                    self.save_unknown_face(frame, (x, y, w, h), self.username)
                    self.unknown_face_saved = True  # Set flag to prevent multiple saves
                    self.status_label.text = 'Unknown face detected and saved'
                    show_popup('Alert', 'Unknown face detected and saved')
            
            # Convert to texture for display
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            
    def verification_success(self):
        self.stop_verification()
        show_popup('Success', 'Face verification successful!')
        self.manager.transition.direction = 'left'
        self.manager.current = 'main'

class ForgotPasswordScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set screen background color
        with self.canvas.before:
            Color(*COLORS['bg_dark'])
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self._update_bg, size=self._update_bg)
        
        # Create the main layout with adjusted padding and spacing
        self.layout = BoxLayout(
            orientation='vertical',
            spacing=15,  # Reduced spacing between elements
            padding=[20, 40, 20, 20]  # Added more top padding (40)
        )

        # Create a spacer at the top to push content down less
        top_spacer = Widget(size_hint_y=0.1)  # Reduced from typical 0.2 or 0.3
        self.layout.add_widget(top_spacer)

        # Title with reduced margin
        self.title_label = Label(
            text='Forgot Password',
            size_hint_y=None,
            height='40dp',  # Slightly reduced height
            font_size='24sp',
            color=COLORS['text_primary'],
            padding=[0, 0]  # Remove any internal padding
        )
        self.layout.add_widget(self.title_label)

        # Add a small spacer after title
        title_spacer = Widget(size_hint_y=0.02)
        self.layout.add_widget(title_spacer)

        # Username input
        self.username_input = TextInput(
            multiline=False,
            hint_text='Username',
            size_hint_y=None,
            height='40dp'
        )
        self.layout.add_widget(self.username_input)

        # Email input with reduced margin
        self.email_input = TextInput(
            multiline=False,
            hint_text='Email',
            size_hint_y=None,
            height='40dp'
        )
        self.layout.add_widget(self.email_input)

        # Add a small spacer before buttons
        button_spacer = Widget(size_hint_y=0.02)
        self.layout.add_widget(button_spacer)

        # Verify and Reset Button
        self.verify_btn = StylizedButton(
            text='Verify and Reset Password',
            primary=True
        )
        self.verify_btn.bind(on_press=self.verify_email_and_reset_password)
        self.layout.add_widget(self.verify_btn)

        # Back Button
        self.back_btn = StylizedButton(
            text='Back',
            primary=False
        )
        self.back_btn.bind(on_press=self.back_to_login)
        self.layout.add_widget(self.back_btn)

        # Add a flexible spacer at the bottom to push everything up
        bottom_spacer = Widget(size_hint_y=0.3)
        self.layout.add_widget(bottom_spacer)

        # Add the layout to the screen
        self.add_widget(self.layout)

    # Rest of the methods remain unchanged
    def _update_bg(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size

    def verify_email_and_reset_password(self, instance):
        username = self.username_input.text
        email = self.email_input.text

        try:
            with open(Config.CREDENTIALS_FILE, 'rb') as f:
                credentials = pickle.load(f)
        except:
            credentials = {}

        if username in credentials:
            if credentials[username].get('email') == email:
                self.show_password_reset_popup()
            else:
                self.show_error_popup("Email does not match our records.")
        else:
            self.show_error_popup("Username does not exist.")

    def show_password_reset_popup(self):
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        # Style the popup background
        with content.canvas.before:
            Color(*COLORS['card_dark'])
            Rectangle(pos=content.pos, size=content.size)
        
        password_input = TextInput(
            multiline=False,
            password=True,
            hint_text='Enter new password',
            size_hint_y=None,
            height='40dp'
        )

        content.add_widget(password_input)

        def do_reset(instance):
            new_password = password_input.text
            if new_password:
                username = self.username_input.text
                try:
                    with open(Config.CREDENTIALS_FILE, 'rb') as f:
                        credentials = pickle.load(f)
                    
                    credentials[username]['password'] = new_password
                    
                    with open(Config.CREDENTIALS_FILE, 'wb') as f:
                        pickle.dump(credentials, f)
                    
                    self.show_success_popup("Password reset successful.")
                    popup.dismiss()
                    self.back_to_login(None)
                except Exception as e:
                    self.show_error_popup(f"Error saving credentials: {str(e)}")
                    popup.dismiss()
            else:
                self.show_error_popup("Password cannot be empty.")
                popup.dismiss()

        reset_btn = StylizedButton(
            text='Reset Password',
            primary=True
        )
        reset_btn.bind(on_press=do_reset)
        content.add_widget(reset_btn)

        popup = Popup(
            title='Reset Password',
            content=content,
            size_hint=(None, None),
            size=('300dp', '200dp'),
            background_color=COLORS['card_dark'],
            title_color=COLORS['text_primary'],
            separator_color=COLORS['primary']
        )
        popup.open()

    def show_error_popup(self, message):
        content = Label(
            text=message,
            color=COLORS['text_primary']
        )
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(None, None),
            size=('300dp', '150dp'),
            background_color=COLORS['card_dark'],
            title_color=COLORS['error'],
            separator_color=COLORS['error']
        )
        popup.open()

    def show_success_popup(self, message):
        content = Label(
            text=message,
            color=COLORS['text_primary']
        )
        popup = Popup(
            title='Success',
            content=content,
            size_hint=(None, None),
            size=('300dp', '150dp'),
            background_color=COLORS['card_dark'],
            title_color=COLORS['primary'],
            separator_color=COLORS['primary']
        )
        popup.open()

    def back_to_login(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = 'login'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        self.face_detector = FaceDetector()

        # Title
        self.layout.add_widget(Label(
            text='Face Recognition System',
            font_size='24sp',
            size_hint_y=None,
            height='50dp'
        ))

        # Camera preview
        self.image = Image(size_hint_y=0.6)
        self.layout.add_widget(self.image)

        # Status label
        self.status_label = Label(
            text='Recognition Status',
            size_hint_y=None,
            height='50dp'
        )
        self.layout.add_widget(self.status_label)

        # Buttons
        buttons = [
            ('Start Recognition', self.start_recognition, True),
            ('Stop Recognition', self.stop_recognition, False),
            ('Show Known Faces', self.show_known_faces, True),
            ('Show Unknown Faces', self.show_unknown_faces, False),
            ('Logout', self.logout, False)
        ]

        for text, callback, primary in buttons:
            btn = StylizedButton(
                text=text,
                primary=primary
            )
            btn.bind(on_press=callback)
            self.layout.add_widget(btn)

        self.add_widget(self.layout)

        self.capture = None
        self.capture_event = None

    def start_recognition(self, instance):
        self.capture = cv2.VideoCapture(0)
        self.capture_event = Clock.schedule_interval(self.update_recognition, 1.0/30.0)
        self.status_label.text = "Recognition Started"

    def stop_recognition(self, instance):
        if self.capture_event:
            self.capture_event.cancel()
        if self.capture:
            self.capture.release()
        self.status_label.text = "Recognition Stopped"
        self.image.texture = None

    def update_recognition(self, dt):
        ret, frame = self.capture.read()
        if ret:
            faces = self.face_detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                found_match = False
                best_similarity = 0
                matched_name = "Unknown"

                for filename in os.listdir(Config.KNOWN_DIR):
                    if filename.endswith('.jpg'):
                        known_path = os.path.join(Config.KNOWN_DIR, filename)
                        similarity = self.face_detector.compare_faces(known_path, face_img)

                        if similarity > best_similarity:
                            best_similarity = similarity
                            matched_name = os.path.splitext(filename)[0]

                if best_similarity >= Config.CONFIDENCE_THRESHOLD:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{matched_name} ({best_similarity:.2f})",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.status_label.text = f"Recognized: {matched_name}"
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Unknown ({best_similarity:.2f})",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.status_label.text = "Unknown Face Detected"

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(Config.UNKNOWN_DIR, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(save_path, face_img)

            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
    
    def show_known_faces(self, instance):
        """Show known faces gallery"""
        self.show_gallery(Config.KNOWN_DIR, "Known Faces")
    
    def show_unknown_faces(self, instance):
        """Show unknown faces gallery"""
        self.show_gallery(Config.UNKNOWN_DIR, "Unknown Faces")
        
    def show_gallery(self, directory, title):
        """Show gallery of faces from specified directory"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Create scrollable grid
        scroll = ScrollView()
        grid = GridLayout(cols=2, spacing=10, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))
        
        # Add images to grid
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Create image layout
                img_layout = BoxLayout(orientation='vertical', 
                                     size_hint_y=None, 
                                     height=250)  
                
                # Load and display image
                image_path = os.path.join(directory, filename)
                img = Image(source=image_path)
                img_layout.add_widget(img)
                
                # Add filename label
                label = Label(text=filename,
                            size_hint_y=None,
                            height=30)
                img_layout.add_widget(label)
                
                # Add delete button
                delete_btn = Button(
                    text="Delete",
                    size_hint_y=None,
                    height=40,
                    background_color=(1, 0.3, 0.3, 1)  
                )
                
                # Create deletion confirmation popup
                def create_delete_popup(img_path, layout):
                    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
                    content.add_widget(Label(text='Are you sure you want to delete this image?'))
                    
                    buttons = BoxLayout(size_hint_y=None, height=40, spacing=10)
                    
                    def confirm_delete(instance):
                        try:
                            os.remove(img_path)  
                            grid.remove_widget(layout) 
                            deletion_popup.dismiss()
                            self.status_label.text = f"Deleted: {os.path.basename(img_path)}"
                        except Exception as e:
                            self.status_label.text = f"Error deleting file: {str(e)}"
                    
                    confirm_btn = Button(text='Confirm', background_color=(1, 0.3, 0.3, 1))
                    cancel_btn = Button(text='Cancel')
                    
                    confirm_btn.bind(on_press=confirm_delete)
                    cancel_btn.bind(on_press=lambda x: deletion_popup.dismiss())
                    
                    buttons.add_widget(confirm_btn)
                    buttons.add_widget(cancel_btn)
                    content.add_widget(buttons)
                    
                    deletion_popup = Popup(
                        title='Confirm Deletion',
                        content=content,
                        size_hint=(0.7, 0.3),
                        auto_dismiss=True
                    )
                    
                    return deletion_popup
                
                # Bind delete button to show confirmation popup
                delete_btn.bind(
                    on_press=lambda btn: create_delete_popup(
                        image_path, img_layout
                    ).open()
                )
                
                img_layout.add_widget(delete_btn)
                grid.add_widget(img_layout)
        
        scroll.add_widget(grid)
        content.add_widget(scroll)
        
        # Add close button
        close_btn = Button(
            text="Close",
            size_hint_y=None,
            height=50
        )
        
        # Create popup
        popup = Popup(
            title=title,
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        content.add_widget(close_btn)
        
        popup.open()
    
    def logout(self, instance):
        """Logout and return to login screen"""
        if self.capture_event:
            self.stop_recognition(None)
        self.manager.transition.direction = 'right'
        self.manager.current = 'login'

class FaceRecognitionApp(App):
    def build(self): 
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(ForgotPasswordScreen(name='forgot_password'))
        sm.add_widget(RegisterScreen(name='register'))
        sm.add_widget(VerifyScreen(name='verify'))
        sm.add_widget(MainScreen(name='main'))
        
        return sm

if __name__ == '__main__':
    FaceRecognitionApp().run()
