import unittest
from flask import json
from app import app  # Ensure this matches the actual filename

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_chat_endpoint(self):
        # Define a test payload (make sure it matches the model's expected input)
        payload = json.dumps({"message": "Hi"})
        response = self.app.post('/chat', headers={"Content-Type": "application/json"}, data=payload)

        # Check the response status code
        self.assertEqual(response.status_code, 200)

        # Check the response content
        response_data = json.loads(response.data)
        print(response_data)  # Optional: Print the response data for debugging

        # Assuming the model returns a non-empty response
        self.assertIn('response', response_data)
        self.assertEqual(response_data['response'], "Hallo")  # Adjust based on your expected response

if __name__ == '__main__':
    unittest.main()
