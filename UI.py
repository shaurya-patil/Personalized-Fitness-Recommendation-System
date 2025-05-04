import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QPushButton, QFrame, QTextEdit, QGridLayout,
                             QMessageBox, QScrollArea, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Matplotlib canvas for embedding in PyQt - INCREASED SIZE
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):  # Increased from 5x4 to 8x6
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.tight_layout()

class PredictionWorker(QThread):
    """Worker thread to handle predictions without freezing the UI"""
    finished = pyqtSignal(float, float, dict, dict)
    
    def __init__(self, health_bot, user_input):
        super().__init__()
        self.health_bot = health_bot
        self.user_input = user_input
        
    def run(self):
        bmi, predicted_time, output, suggestions = self.health_bot.predict(self.user_input)
        self.finished.emit(bmi, predicted_time, output, suggestions)

class HealthChatBot:
    def __init__(self, data):
        self.raw_data = data.copy()
        self.nutrition_model = None
        self.time_model = None  # New model for time prediction
        self.scaler = StandardScaler()
        self.time_scaler = StandardScaler()  # Separate scaler for time model
        self.label_encoder = LabelEncoder()
        
        # Columns for nutrition model (without time to target)
        self.input_cols = ['Body Mass Index (BMI)', 'Body Weight (kg)', 'Predicted Target Weight (kg)',
                           'Body Height (cm)', 'Disease']
        
        # Columns for time prediction model
        self.time_input_cols = ['Body Mass Index (BMI)', 'Body Weight (kg)', 'Predicted Target Weight (kg)',
                               'Body Height (cm)', 'Disease']
        
        self.num_cols = ['Body Mass Index (BMI)', 'Body Weight (kg)', 'Predicted Target Weight (kg)',
                         'Body Height (cm)']
        
        self.target_cols = ['Protein', 'Sugar', 'Sodium', 'Calories', 'Carbohydrates', 'Fiber', 'Fat',
                            'Breakfast Calories', 'Breakfast Protein', 'Breakfast Carbohydrates', 'Breakfast Fats',
                            'Lunch Calories', 'Lunch Protein', 'Lunch Carbohydrates', 'Lunch Fats',
                            'Dinner Calories', 'Dinner Protein.1', 'Dinner Carbohydrates.1', 'Dinner Fats',
                            'Snacks Calories', 'Snacks Protein', 'Snacks Carbohydrates', 'Snacks Fats']
        
        self.suggestion_cols = ['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion', 'suggestions']
        self._prepare_data()
        
        # Cache for disease encodings to avoid repeated transformations
        self.disease_encodings = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))

    def _prepare_data(self):
        # Check if models are already trained and saved
        nutrition_model_path = "health_model.joblib"
        time_model_path = "time_model.joblib"
        scaler_path = "scaler.joblib"
        time_scaler_path = "time_scaler.joblib"
        encoder_path = "label_encoder.joblib"
        suggestion_data_path = "suggestion_data.pkl"
        
        if all(os.path.exists(p) for p in [nutrition_model_path, time_model_path, scaler_path, time_scaler_path, encoder_path, suggestion_data_path]):
            # Load pre-trained models and preprocessors
            self.nutrition_model = joblib.load(nutrition_model_path)
            self.time_model = joblib.load(time_model_path)
            self.scaler = joblib.load(scaler_path)
            self.time_scaler = joblib.load(time_scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.suggestion_data = pd.read_pickle(suggestion_data_path)
        else:
            # Train new models
            df = self.raw_data.dropna(subset=self.target_cols).copy()
            
            # Process disease column
            df['Disease'] = df['Disease'].astype(str).fillna("None")
            df['Disease'] = self.label_encoder.fit_transform(df['Disease'])
            
            # Scale numerical columns
            df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
            
            # Prepare X and y for nutrition model
            X_nutrition = df[self.input_cols + ['Predicted Time to Target (wks)']]
            y_nutrition = df[self.target_cols]

            # Train nutrition model (MultiOutputRegressor with RandomForest)
            self.nutrition_model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            )
            self.nutrition_model.fit(X_nutrition, y_nutrition)
            
            # Train time prediction model
            # Make sure we have data for time prediction
            time_df = df.dropna(subset=['Predicted Time to Target (wks)']).copy()
            
            # Scale numerical inputs for time model
            X_time = time_df[self.time_input_cols]
            y_time = time_df['Predicted Time to Target (wks)']
            
            # Train time model (RandomForestRegressor)
            self.time_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.time_model.fit(X_time, y_time)

            # Store suggestion data
            self.suggestion_data = df[self.input_cols + ['Predicted Time to Target (wks)'] + self.suggestion_cols].copy()
            
            # Save models and preprocessors for future use
            joblib.dump(self.nutrition_model, nutrition_model_path)
            joblib.dump(self.time_model, time_model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.time_scaler, time_scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            self.suggestion_data.to_pickle(suggestion_data_path)

    def predict(self, user_input):
        # Calculate BMI
        bmi = round(user_input['weight'] / ((user_input['height'] / 100) ** 2), 2)
        
        # Get disease encoding using the cache
        disease = user_input.get('disease', 'None')
        disease_encoded = self.disease_encodings.get(disease, 0)
        
        # Create input DataFrame for time prediction
        time_input_data = pd.DataFrame([{
            'Body Mass Index (BMI)': bmi,
            'Body Weight (kg)': user_input['weight'],
            'Predicted Target Weight (kg)': user_input['target_weight'],
            'Body Height (cm)': user_input['height'],
            'Disease': disease_encoded
        }])
        
        # Scale time input data
        time_input_scaled = self.scaler.transform(time_input_data[self.num_cols])
        time_input_data[self.num_cols] = time_input_scaled
        
        # Predict time to target
        predicted_time = max(1, self.time_model.predict(time_input_data)[0])  # Ensure positive value
        
        # Now create input for nutrition model including predicted time
        nutrition_input_data = pd.DataFrame([{
            'Body Mass Index (BMI)': bmi,
            'Body Weight (kg)': user_input['weight'],
            'Predicted Target Weight (kg)': user_input['target_weight'],
            'Body Height (cm)': user_input['height'],
            'Disease': disease_encoded,
            'Predicted Time to Target (wks)': predicted_time
        }])

        # Scale nutrition input data (only numerical columns)
        nutrition_input_data[self.num_cols] = time_input_scaled
        
        # Make predictions for nutrition
        preds = self.nutrition_model.predict(nutrition_input_data)[0]
        output = dict(zip(self.target_cols, preds))
        
        # Add predicted time to user input for suggestion finding
        user_input_with_time = time_input_data.copy()
        user_input_with_time['Predicted Time to Target (wks)'] = predicted_time
        
        # Get closest suggestion
        closest = self.find_closest_suggestion(user_input_with_time.iloc[0])
        suggestions = {}
        for col in self.suggestion_cols:
            if col in closest:
                suggestions[col] = closest[col]
                
        return bmi, predicted_time, output, suggestions

    def find_closest_suggestion(self, user_row):
        """Optimized version using vectorized operations"""
        # Calculate distances all at once using vectorization
        comparison_cols = self.input_cols + ['Predicted Time to Target (wks)']
        # Ensure the same columns are used for comparison
        user_values = user_row[comparison_cols]
        suggestion_values = self.suggestion_data[comparison_cols]
        
        distances = ((suggestion_values - user_values) ** 2).sum(axis=1)
        # Get the index of the minimum distance
        closest_idx = distances.idxmin()
        # Return the row with that index
        return self.suggestion_data.loc[closest_idx]


class HealthChatBotUI(QMainWindow):
    def __init__(self, health_bot):
        super().__init__()
        self.health_bot = health_bot
        self.setWindowTitle("Health Recommendation System")
        self.setMinimumSize(1000, 800)  # Increased window size for larger chart
        
        self.init_ui()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Header
        header = QLabel("Health Recommendation System")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Description
        description = QLabel("Enter your details below to get personalized nutrition recommendations")
        description.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(description)
        
        # Form area
        form_frame = QFrame()
        form_frame.setFrameShape(QFrame.StyledPanel)
        form_layout = QGridLayout()
        form_frame.setLayout(form_layout)
        
        # Input fields
        # Age
        form_layout.addWidget(QLabel("Age:"), 0, 0)
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Enter your age")
        form_layout.addWidget(self.age_input, 0, 1)
        
        # Weight
        form_layout.addWidget(QLabel("Weight (kg):"), 1, 0)
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("Enter your weight in kg")
        form_layout.addWidget(self.weight_input, 1, 1)
        
        # Height
        form_layout.addWidget(QLabel("Height (cm):"), 2, 0)
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Enter your height in cm")
        form_layout.addWidget(self.height_input, 2, 1)
        
        # Target Weight
        form_layout.addWidget(QLabel("Target Weight (kg):"), 3, 0)
        self.target_weight_input = QLineEdit()
        self.target_weight_input.setPlaceholderText("Enter your target weight in kg")
        form_layout.addWidget(self.target_weight_input, 3, 1)
        
        # Disease
        form_layout.addWidget(QLabel("Do you have any disease?"), 4, 0)
        self.disease_dropdown = QComboBox()
        self.disease_dropdown.addItem("None")
        # Add diseases from the label encoder classes
        for disease in self.health_bot.label_encoder.classes_:
            if disease != "None":
                self.disease_dropdown.addItem(disease)
        form_layout.addWidget(self.disease_dropdown, 4, 1)
        
        # Submit button
        self.submit_button = QPushButton("Get Recommendations")
        self.submit_button.clicked.connect(self.process_input)
        form_layout.addWidget(self.submit_button, 5, 0, 1, 2)
        
        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        form_layout.addWidget(self.progress_bar, 6, 0, 1, 2)
        
        main_layout.addWidget(form_frame)
        
        # Results area with scroll
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        results_widget.setLayout(self.results_layout)
        results_scroll.setWidget(results_widget)
        
        # Initial results widgets
        self.bmi_label = QLabel()
        self.bmi_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(self.bmi_label)
        
        # Time to target label
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(self.time_label)
        
        # Add pie chart header and widget
        self.chart_header = QLabel("📊 Macronutrient Distribution:")
        self.chart_header.setFont(QFont("Arial", 14, QFont.Bold))  # Increased font size
        self.results_layout.addWidget(self.chart_header)
        
        # Add pie chart canvas - INCREASED SIZE
        self.chart_canvas = MplCanvas(self, width=8, height=6)  # Increased from 5x4
        self.chart_canvas.setMinimumHeight(400)  # Set minimum height to ensure visibility
        self.results_layout.addWidget(self.chart_canvas)
        
        self.nutrition_header = QLabel("📋 Nutritional Recommendations:")
        self.nutrition_header.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(self.nutrition_header)
        
        self.calories_label = QLabel()
        self.macros_label = QLabel()
        self.results_layout.addWidget(self.calories_label)
        self.results_layout.addWidget(self.macros_label)
        
        self.meals_header = QLabel("🍽️ Meal-wise Suggestions:")
        self.meals_header.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(self.meals_header)
        
        self.breakfast_label = QLabel()
        self.lunch_label = QLabel()
        self.dinner_label = QLabel()
        self.snacks_label = QLabel()
        
        self.results_layout.addWidget(self.breakfast_label)
        self.results_layout.addWidget(self.lunch_label)
        self.results_layout.addWidget(self.dinner_label)
        self.results_layout.addWidget(self.snacks_label)
        
        self.suggestions_header = QLabel("📌 Meal Suggestions:")
        self.suggestions_header.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(self.suggestions_header)
        
        self.breakfast_suggestion = QTextEdit()
        self.breakfast_suggestion.setReadOnly(True)
        self.breakfast_suggestion.setMaximumHeight(80)
        
        self.lunch_suggestion = QTextEdit()
        self.lunch_suggestion.setReadOnly(True)
        self.lunch_suggestion.setMaximumHeight(80)
        
        self.dinner_suggestion = QTextEdit()
        self.dinner_suggestion.setReadOnly(True)
        self.dinner_suggestion.setMaximumHeight(80)
        
        self.snack_suggestion = QTextEdit()
        self.snack_suggestion.setReadOnly(True)
        self.snack_suggestion.setMaximumHeight(80)
        
        self.results_layout.addWidget(QLabel("Breakfast suggestion:"))
        self.results_layout.addWidget(self.breakfast_suggestion)
        self.results_layout.addWidget(QLabel("Lunch suggestion:"))
        self.results_layout.addWidget(self.lunch_suggestion)
        self.results_layout.addWidget(QLabel("Dinner suggestion:"))
        self.results_layout.addWidget(self.dinner_suggestion)
        self.results_layout.addWidget(QLabel("Snack suggestion:"))
        self.results_layout.addWidget(self.snack_suggestion)
        
        # Hide results initially
        self.hide_results()
        
        main_layout.addWidget(results_scroll)
        
    def hide_results(self):
        self.bmi_label.hide()
        self.time_label.hide()
        self.chart_header.hide()  # Hide chart header
        self.chart_canvas.hide()  # Hide chart canvas
        self.nutrition_header.hide()
        self.calories_label.hide()
        self.macros_label.hide()
        self.meals_header.hide()
        self.breakfast_label.hide()
        self.lunch_label.hide()
        self.dinner_label.hide()
        self.snacks_label.hide()
        self.suggestions_header.hide()
        self.breakfast_suggestion.hide()
        self.lunch_suggestion.hide()
        self.dinner_suggestion.hide()
        self.snack_suggestion.hide()
        
    def show_results(self):
        self.bmi_label.show()
        self.time_label.show()
        self.chart_header.show()  # Show chart header
        self.chart_canvas.show()  # Show chart canvas
        self.nutrition_header.show()
        self.calories_label.show()
        self.macros_label.show()
        self.meals_header.show()
        self.breakfast_label.show()
        self.lunch_label.show()
        self.dinner_label.show()
        self.snacks_label.show()
        self.suggestions_header.show()
        self.breakfast_suggestion.show()
        self.lunch_suggestion.show()
        self.dinner_suggestion.show()
        self.snack_suggestion.show()
        
    def update_pie_chart(self, protein, carbs, fat):
        # Clear the previous chart
        self.chart_canvas.axes.clear()
        
        # Data for pie chart
        labels = ['Protein', 'Carbs', 'Fat']
        sizes = [protein, carbs, fat]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)  # explode the 1st slice (Protein)
        
        # Create pie chart with larger text and elements
        self.chart_canvas.axes.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90, 
                textprops={'fontsize': 12, 'weight': 'bold'})
        self.chart_canvas.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Set title with larger font
        self.chart_canvas.axes.set_title('Macronutrient Distribution', fontsize=16, fontweight='bold')
        
        # Add more space around the pie chart
        self.chart_canvas.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Refresh canvas
        self.chart_canvas.draw()
        
    def process_input(self):
        try:
            # Get input values
            age = int(self.age_input.text())
            weight = float(self.weight_input.text())
            height = float(self.height_input.text())
            target_weight = float(self.target_weight_input.text())
            disease = self.disease_dropdown.currentText()
            
            # Validate inputs
            if age <= 0 or weight <= 0 or height <= 0 or target_weight <= 0:
                QMessageBox.warning(self, "Input Error", "All values must be positive numbers")
                return
            
            user_input = {
                'weight': weight,
                'height': height,
                'target_weight': target_weight,
                'disease': disease if disease != "None" else "None"
            }
            
            # Show loading indicator
            self.submit_button.setEnabled(False)
            self.submit_button.setText("Processing...")
            self.progress_bar.show()
            
            # Create worker thread for prediction
            self.prediction_thread = PredictionWorker(self.health_bot, user_input)
            self.prediction_thread.finished.connect(self.update_results)
            self.prediction_thread.start()
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Please enter valid numerical values: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def update_results(self, bmi, predicted_time, output, suggestions):
        # Update UI with results
        self.bmi_label.setText(f"Your BMI: {bmi:.2f}")
        self.time_label.setText(f"Predicted Time to Target: {predicted_time:.1f} weeks")
        
        # Update pie chart with macronutrient data
        protein = output['Protein']
        carbs = output['Carbohydrates']
        fat = output['Fat']
        self.update_pie_chart(protein, carbs, fat)
        
        self.calories_label.setText(f"Total Calories: {output['Calories']:.2f} kcal")
        self.macros_label.setText(f"Protein: {output['Protein']:.2f}g | Carbs: {output['Carbohydrates']:.2f}g | Fat: {output['Fat']:.2f}g")
        
        self.breakfast_label.setText(
            f"Breakfast: {output['Breakfast Calories']:.0f} kcal | {output['Breakfast Protein']:.1f}g protein | "
            f"{output['Breakfast Carbohydrates']:.1f}g carbs | {output['Breakfast Fats']:.1f}g fat"
        )
        
        self.lunch_label.setText(
            f"Lunch: {output['Lunch Calories']:.0f} kcal | {output['Lunch Protein']:.1f}g protein | "
            f"{output['Lunch Carbohydrates']:.1f}g carbs | {output['Lunch Fats']:.1f}g fat"
        )
        
        self.dinner_label.setText(
            f"Dinner: {output['Dinner Calories']:.0f} kcal | {output['Dinner Protein.1']:.1f}g protein | "
            f"{output['Dinner Carbohydrates.1']:.1f}g carbs | {output['Dinner Fats']:.1f}g fat"
        )
        
        self.snacks_label.setText(
            f"Snacks: {output['Snacks Calories']:.0f} kcal | {output['Snacks Protein']:.1f}g protein | "
            f"{output['Snacks Carbohydrates']:.1f}g carbs | {output['Snacks Fats']:.1f}g fat"
        )
        
        # Update suggestions
        self.breakfast_suggestion.setText(suggestions.get('Breakfast Suggestion', 'No suggestion available'))
        self.lunch_suggestion.setText(suggestions.get('Lunch Suggestion', 'No suggestion available'))
        self.dinner_suggestion.setText(suggestions.get('Dinner Suggestion', 'No suggestion available'))
        self.snack_suggestion.setText(suggestions.get('Snack Suggestion', 'No suggestion available'))
        
        # Show results
        self.show_results()
        
        # Reset UI
        self.submit_button.setEnabled(True)
        self.submit_button.setText("Get Recommendations")
        self.progress_bar.hide()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            QApplication.quit()


if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv("dataset/FULL_DATASET.csv")
        df.columns = df.columns.str.strip()
        
        # Create the health bot
        health_bot = HealthChatBot(df)
        
        # Create and show UI
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Modern style
        window = HealthChatBotUI(health_bot)
        window.show()
        sys.exit(app.exec_())
        
    except FileNotFoundError:
        print("Error: Dataset file not found. Make sure 'dataset/FULL_DATASET.csv' exists.")
    except Exception as e:
        print(f"Error initializing application: {str(e)}")