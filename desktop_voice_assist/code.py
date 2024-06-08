import mediapipe as mp
import cv2
import datetime
import webbrowser
import os
import requests
import pyttsx3
import warnings
import math
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from dateutil.parser import parse
import calendar
import subprocess  # Added import for subprocess

class VoiceAssistant:

    def _init_(self):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Initialize GPT-2 model and tokenizer
        model_name = "gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.cap = cv2.VideoCapture(0)  # Use the default camera (0)


    def speak(self, text):
        """Function to convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def get_audio(self):
        """Function to capture audio input"""
        try:
            audio = self.record_audio()
            print("Recognizing...")
            query = self.recognize_audio(audio).lower()
            print(f"You said: {query}")
            return query
        except Exception as e:
            print(f"Error with audio input: {e}")
            return ""

    def record_audio(self):
        """Function to record audio"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source)
            return audio

    def recognize_audio(self, audio):
        """Function to recognize audio"""
        return self.recognizer.recognize_google(audio)

    def open_website(self, query):
        """Function to open a website based on the query"""
        if "open youtube" in query:
            webbrowser.open("https://www.youtube.com")
        elif "open google" in query:
            webbrowser.open("https://www.google.com")
        elif "open wikipedia" in query:
            webbrowser.open("https://www.wikipedia.org")
        else:
            self.speak("Sorry, I couldn't recognize the website.")

    def calculator(self, query):
        """Function to perform calculations based on spoken instructions."""
        try:
            # Create a mapping from word to numeric values
            word_to_number = {
                'zero': 0,
                'one': 1,
                'two': 2,
                'three': 3,
                'four': 4,
                'five': 5,
                'six': 6,
                'seven': 7,
                'eight': 8,
                'nine': 9,
                'ten': 10,
            }

            # Split the query into words
            words = query.split()

            # Extract elements from the query
            if len(words) != 3:
                self.speak("Sorry, I couldn't understand the query format. Please use 'number1 operation number2'.")
                return

            number1_word, operation, number2 = map(str.strip, words)

            # Convert number1_word to numeric value
            if number1_word in word_to_number:
                number1 = word_to_number[number1_word]
            else:
                self.speak(f"Sorry, I couldn't recognize '{number1_word}' as a valid number word.")
                return

            # Check if number2 is a valid integer
            if not number2.isdigit():
                self.speak(f"Sorry, '{number2}' is not a valid integer.")
                return

            number2 = int(number2)  # Convert number2 to an integer

            # Perform the calculation based on the operation
            if operation == "addition":
                result = number1 + number2
            elif operation == "subtraction":
                result = number1 - number2
            elif operation == "multiplication":
                result = number1 * number2
            elif operation == "division":
                if number2 != 0:
                    result = number1 / number2
                else:
                    self.speak("Division by zero is not allowed.")
                    return
            else:
                self.speak("Sorry, I couldn't recognize the operation.")
                return

            self.speak(f"The result of {query} is {result}")
        except Exception as e:
            self.speak("Sorry, I couldn't perform the calculation. Please try again.")

    def system_calendar(self, query):
        """Function to access the system's calendar."""
        try:
            if "view calendar" in query:
                year, month = datetime.datetime.now().year, datetime.datetime.now().month
                cal = calendar.month(year, month)
                self.speak(f"Here is the calendar for {calendar.month_name[month]} {year}:\n{cal}")
            elif "view next month's calendar" in query:
                year, month = datetime.datetime.now().year, datetime.datetime.now().month + 1
                if month > 12:
                    year += 1
                    month = 1
                cal = calendar.month(year, month)
                self.speak(f"Here is the calendar for {calendar.month_name[month]} {year}:\n{cal}")
            else:
                self.speak("Sorry, I couldn't recognize the calendar command.")
        except Exception as e:
            self.speak("Sorry, there was an error accessing the calendar.")

    def set_reminder(self, query):
        """Function to set a reminder using the system's calendar."""
        try:
            query = query.replace("set a reminder for", "")
            reminder_time = parse(query)
            subprocess.Popen(["/usr/bin/gnome-calendar", "--reminder", f"--time={reminder_time.strftime('%H:%M')}",
                              f"--date={reminder_time.strftime('%Y-%m-%d')}"])
            self.speak(f"Reminder set for {reminder_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            self.speak("The remainder is set for tomorrow ")

    def set_appointment(self, query):
        """Function to set an appointment using the system's calendar."""
        try:
            query = query.replace("set an appointment for", "")
            appointment_time = parse(query)
            subprocess.Popen(["/usr/bin/gnome-calendar", "--add-event",
                              f"--time={appointment_time.strftime('%H:%M')}",
                              f"--date={appointment_time.strftime('%Y-%m-%d')}"])
            self.speak(f"Appointment set for {appointment_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            self.speak("Sorry, I couldn't set the appointment. Please try again.")

    def add_event_to_calendar(self, query):
        """Function to add an event to the system's calendar."""
        try:
            query = query.replace("add an event to calendar", "")
            event_time = parse(query)
            subprocess.Popen(["/usr/bin/gnome-calendar", "--add-event",
                              f"--time={event_time.strftime('%H:%M')}",
                              f"--date={event_time.strftime('%Y-%m-%d')}"])
            self.speak(f"Event added to calendar: {event_time.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            self.speak("Sorry, I couldn't add the event to the calendar. Please try again.")

    def perform_basic_tasks(self, query):
        """Function to perform basic tasks based on the query"""
        if "what's the time" in query or "tell me the time" in query:
            current_time = datetime.datetime.now().strftime("%H:%M")
            self.speak(f"The current time is {current_time}")
        elif "open notepad" in query:
            os.startfile("notepad.exe")
        elif "search" in query:
            search_query = query.replace("search", "")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        elif "weather" in query:
            city = query.split("weather in ")[-1]
            api_key = 'YOUR_API_KEY'  # Replace with your OpenWeatherMap API key
            self.get_weather(api_key, city)
        elif "calculate" in query:
            self.calculator(query)
        elif "set a reminder for" in query:
            self.set_reminder(query)
        elif "set an appointment for" in query:
            self.set_appointment(query)
        elif "add an event to calendar" in query:
            self.add_event_to_calendar(query)
        elif "calendar" in query:
            self.system_calendar(query)
        else:
            self.speak("Sorry, I couldn't recognize the command.")

    def get_weather(self, api_key, city):
        """Function to get weather information for a specific city."""
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "imperial"}  # You can adjust the units as needed

        try:
            response = requests.get(base_url, params=params)
            weather_data = response.json()

            if response.status_code == 200:
                weather_description = weather_data["weather"][0]["main"]
                temperature = weather_data["main"]["temp"]
                self.speak(f"The weather in {city} is {weather_description}. "
                           f"The temperature is {temperature} degrees Fahrenheit.")
            else:
                self.speak("Sorry, I couldn't fetch the weather information. Please try again later.")

        except Exception as e:
            print(f"Error fetching weather information: {e}")
            self.speak("Sorry, there was an error fetching the weather information.")




    def process_volume_control(self):
        """Function to process hand gesture for volume control"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mp_drawing = mp.solutions.drawing_utils

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volMin, volMax = volume.GetVolumeRange()[:2]

        stop_control = False

        while not stop_control:
            success, img = self.cap.read()  # Use self.cap here to access the VideoCapture

            if not success:
                continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            lmList = []
            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    for id, lm in enumerate(handlandmark.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mp_drawing.draw_landmarks(img, handlandmark, mp_hands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [30, 350], [volMin, volMax])
                volbar = np.interp(length, [30, 350], [400, 150])
                volper = np.interp(length, [30, 350], [0, 100])

                print(vol, int(length))
                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
                cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

            cv2.imshow('Image', img)

            key = cv2.waitKey(1) & 0xff
            if key == ord(' '):  # Press spacebar to stop
                stop_control = True
            elif key == ord('q'):  # Press 'q' to exit
                stop_control = True

            # Check for voice command "quit"
            query = self.get_audio()
            if "quit" in query:
                print("Quitting volume control based on voice command.")
                return

        self.cap.release()
        cv2.destroyAllWindows()



    def run(self):
        self.speak("Hello! How can I assist you today?")
        while True:
            query = self.get_audio()

            if "exit" in query or "quit" in query:
                self.speak("Goodbye!")
                break
            elif "open" in query:
                self.speak("Sure, which website would you like to open?")
                website_query = self.get_audio().lower()
                self.open_website(website_query)
            elif "calculator" in query:
                self.speak("Sure, let's calculate. Please specify the calculation, for example, '5 addition 3'.")
                calculation_query = self.get_audio().lower()
                self.calculator(calculation_query)
            elif "calendar" in query:
                self.speak("Sure, let's access the calendar. Would you like to view the calendar or view next month's calendar?")
                calendar_action = self.get_audio().lower()
                self.system_calendar(calendar_action)
            elif "reminder" in query:
                self.speak("Sure, let's set a reminder. Please specify the reminder time and date.")
                reminder_query = self.get_audio().lower()
                self.set_reminder(reminder_query)
            elif "appointment" in query:
                self.speak("Sure, let's set an appointment. Please specify the appointment time and date.")
                appointment_query = self.get_audio().lower()
                self.set_appointment(appointment_query)
            elif "event" in query:
                self.speak("Sure, let's add an event to the calendar. Please specify the event time and date.")
                event_query = self.get_audio().lower()
                self.add_event_to_calendar(event_query)
            elif "chat" in query:
                self.speak("Sure, let's chat. Ask me anything.")
                while True:
                    user_query = self.get_audio()
                    if "exit" in user_query or "quit" in user_query:
                        self.speak("Ending the chat. How can I assist you further?")
                        break
                    if "how are you" in user_query:
                        self.speak("i am fine, how are you?")
            elif "volume" in query:
                self.speak("Turning on volume control. Perform the hand gesture to control the volume.")
                self.process_volume_control()
            else:
                self.perform_basic_tasks(query)

if _name_ == "_main_":
    assistant = VoiceAssistant()
    assistant.run()