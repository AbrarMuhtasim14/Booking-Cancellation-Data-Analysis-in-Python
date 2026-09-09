"""
Unit Tests for Hotel Booking Cancellation Model & Inferences
Uses standard unittest library for zero-dependency test execution.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class TestHotelCancellationModel(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
        self.encoders_path = os.path.join(MODELS_DIR, "label_encoders.joblib")

    def test_model_and_encoders_exist(self):
        self.assertTrue(os.path.exists(self.model_path), "Model file random_forest_model.joblib is missing!")
        self.assertTrue(os.path.exists(self.encoders_path), "Encoders file label_encoders.joblib is missing!")

    def test_model_prediction_shape_and_range(self):
        model = joblib.load(self.model_path)
        encoders = joblib.load(self.encoders_path)

        test_input = {
            'lead_time': [45, 300],
            'arrival_date_month': ['July', 'August'],
            'country': ['PRT', 'PRT'],
            'market_segment': ['Direct', 'Online TA'],
            'distribution_channel': ['Direct', 'TA/TO'],
            'reserved_room_type': ['A', 'A'],
            'deposit_type': ['No Deposit', 'Non Refund'],
            'customer_type': ['Transient', 'Transient']
        }

        df = pd.DataFrame(test_input)

        for col in encoders:
            df[col] = encoders[col].transform(df[col])

        preds = model.predict(df)
        probas = model.predict_proba(df)

        self.assertEqual(len(preds), 2)
        self.assertEqual(probas.shape, (2, 2))
        self.assertTrue(np.all((probas >= 0.0) & (probas <= 1.0)))
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))

    def test_high_lead_time_higher_risk(self):
        model = joblib.load(self.model_path)
        encoders = joblib.load(self.encoders_path)

        low_lead = pd.DataFrame([{
            'lead_time': 5,
            'arrival_date_month': encoders['arrival_date_month'].transform(['May'])[0],
            'country': encoders['country'].transform(['PRT'])[0],
            'market_segment': encoders['market_segment'].transform(['Direct'])[0],
            'distribution_channel': encoders['distribution_channel'].transform(['Direct'])[0],
            'reserved_room_type': encoders['reserved_room_type'].transform(['A'])[0],
            'deposit_type': encoders['deposit_type'].transform(['No Deposit'])[0],
            'customer_type': encoders['customer_type'].transform(['Transient'])[0]
        }])

        high_lead = pd.DataFrame([{
            'lead_time': 350,
            'arrival_date_month': encoders['arrival_date_month'].transform(['August'])[0],
            'country': encoders['country'].transform(['PRT'])[0],
            'market_segment': encoders['market_segment'].transform(['Online TA'])[0],
            'distribution_channel': encoders['distribution_channel'].transform(['TA/TO'])[0],
            'reserved_room_type': encoders['reserved_room_type'].transform(['A'])[0],
            'deposit_type': encoders['deposit_type'].transform(['No Deposit'])[0],
            'customer_type': encoders['customer_type'].transform(['Transient'])[0]
        }])

        p_low = model.predict_proba(low_lead)[0][1]
        p_high = model.predict_proba(high_lead)[0][1]

        self.assertGreater(
            p_high, p_low,
            f"Expected higher cancellation probability for long lead time ({p_high:.2f} vs {p_low:.2f})"
        )


if __name__ == '__main__':
    unittest.main()
