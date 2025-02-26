import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=100):
    # Indian names
    first_names = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayush", "Sai",
        "Ananya", "Diya", "Saanvi", "Aanya", "Aadhya", "Ishita", "Kavya", "Khushi"
    ]
    
    last_names = [
        "Patel", "Kumar", "Singh", "Sharma", "Verma", "Gupta", "Shah", "Reddy",
        "Kapoor", "Malhotra", "Joshi", "Chopra", "Mehta", "Sinha", "Rao", "Desai"
    ]
    
    # Generate random data
    data = {
        'suspect_id': [f'S{str(i).zfill(4)}' for i in range(1, num_records + 1)],
        'name': [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" 
                for _ in range(num_records)],
        'age': np.random.randint(18, 65, num_records),
        'gender': np.random.choice(['M', 'F'], num_records),
        'height': np.random.normal(170, 10, num_records).round(1),
        'weight': np.random.normal(70, 12, num_records).round(1),
        'gait_pattern': np.random.choice([
            'Normal', 'Limping', 'Shuffling', 'Heavy-stepped', 'Quick-paced',
            'Asymmetric', 'Distinctive swagger', 'Toe-walking'
        ], num_records),
        'criminal_history': np.random.choice([
            'Theft', 'Burglary', 'Robbery', 'Assault', 'None'
        ], num_records, p=[0.25, 0.25, 0.2, 0.1, 0.2]),
        'last_seen': [
            (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(num_records)
        ],
        'gait_signature': [
            {
                'stride_length_avg': np.random.normal(75, 10),
                'step_width_avg': np.random.normal(15, 3),
                'cadence': np.random.normal(110, 10),
                'gait_symmetry': np.random.uniform(0.8, 1.0),
                'distinctive_features': np.random.choice([
                    'Pronated feet', 'Supinated feet', 'Knee valgus', 'Knee varus', 'None'
                ])
            } for _ in range(num_records)
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def save_database():
    df = generate_synthetic_data()
    df.to_csv('criminal_gait_database.csv', index=False)
    return df

if __name__ == "__main__":
    save_database() 