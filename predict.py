#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple prediction script for airline passenger satisfaction model
"""

import pickle
import json

def load_model(model_file):
    """
    Load the trained model and vectorizer
    """
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    dv = model_data['vectorizer']
    model = model_data['model']
    
    return dv, model

def predict_single_passenger(customer, dv, model):
    """
    Predict satisfaction for a single passenger
    """
    # Transform input data
    X = dv.transform([customer])
    
    # Predict probability
    y_pred = model.predict_proba(X)[0, 1]
    
    # Predict class
    satisfaction = y_pred >= 0.5
    
    return y_pred, satisfaction

def main():
    """
    Main function: Load model and make predictions
    """
    # Model file path
    model_file = 'output/best_model.pkl'
    
    try:
        # 1. Load model
        print("Loading model...")
        dv, model = load_model(model_file)
        print("✅ Model loaded successfully\n")
        
        # 2. Example passenger data
        example_customer = {
            'age': 45,
            'gender': 'male',
            'customer_type': 'loyal_customer',
            'type_of_travel': 'business_travel',
            'class': 'business',
            'flight_distance': 2500,
            'inflight_wifi_service': 3,
            'departure/arrival_time_convenient': 4,
            'ease_of_online_booking': 3,
            'gate_location': 4,
            'food_and_drink': 4,
            'online_boarding': 4,
            'seat_comfort': 4,
            'inflight_entertainment': 4,
            'on-board_service': 4,
            'leg_room_service': 4,
            'baggage_handling': 4,
            'checkin_service': 4,
            'inflight_service': 4,
            'cleanliness': 4,
            'departure_delay_in_minutes': 0,
            'arrival_delay_in_minutes': 0
        }
        
        # 3. Display input data
        print("📋 Passenger Information:")
        print("=" * 40)
        for key, value in example_customer.items():
            print(f"{key:30}: {value}")
        print("=" * 40)
        
        # 4. Make prediction
        print("\n🔮 Making prediction...")
        probability, satisfied = predict_single_passenger(example_customer, dv, model)
        
        # 5. Print results
        print("\n🎯 PREDICTION RESULT:")
        print("=" * 40)
        print(f"Satisfaction Probability: {probability:.3f}")
        print(f"Predicted Class: {'Satisfied' if satisfied else 'Neutral/Dissatisfied'}")
        print(f"Threshold: 0.5")
        print("=" * 40)
        
        # 6. Interpretation
        print("\n📊 Interpretation:")
        if satisfied:
            print("✅ This passenger is predicted to be SATISFIED with the flight")
            print(f"   (Probability: {probability:.1%})")
        else:
            print("⚠️  This passenger is predicted to be NEUTRAL or DISSATISFIED")
            print(f"   (Probability of satisfaction: {probability:.1%})")
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_file}")
        print("Please run train.py first to train and save the model.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()