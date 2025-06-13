import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime

start_date = datetime.strptime("2024-01-01", "%Y-%m-%d").date()
end_date = datetime.strptime("2024-12-31", "%Y-%m-%d").date()
fake = Faker('en_US')  # Changed to US locale
np.random.seed(42)
random.seed(42)

# Parameters
n_records = 1000
regions = ['New York', 'California', 'Texas', 'Florida', 'Illinois', 
           'Pennsylvania', 'Ohio', 'Georgia']  # US states
services = ['Account Opening', 'Loan Services', 'Investment Services', 
            'Online Banking', 'Branch Services']  # English service names
n_questions = 6

# Thematic features
thematic_flags = [
    'WasQueueLong',
    'WasStaffPolite',
    'UsedMobileBanking',
    'IssueResolved',
    'WaitTimeCategory'
]

# Wait time categories in English
wait_time_categories = ['short', 'medium', 'long']

# Regenerate data with new features
data = []
for _ in range(n_records):
    client_company = fake.company()
    division = random.choice(['Headquarters', 'Regional Branch', 'Online Division'])
    region = random.choice(regions)

    service_type = random.choice(services)
    service_cost = round(np.random.normal(3000, 1000), 2)

    customer_name = fake.name()
    customer_contact = fake.phone_number()
    customer_address = fake.address().replace('\n', ', ')

    date_interview = fake.date_between(start_date=start_date, end_date=end_date)
    invoice_date = date_interview
    close_date = fake.date_between(start_date=date_interview, end_date=end_date)

    scores = np.clip(np.random.normal(loc=7.5, scale=2.0, size=n_questions).round(), 1, 10).astype(int)
    average_score = scores.mean()

    if average_score >= 9:
        nps_label = 'Promoter'
    elif average_score >= 7:
        nps_label = 'Passive'
    else:
        nps_label = 'Detractor'

    # Thematic features
    was_queue_long = random.choice([True, False])
    was_staff_polite = random.choices([True, False], weights=[0.8, 0.2])[0]
    used_mobile_banking = random.choice([True, False])
    issue_resolved = random.choices([True, False], weights=[0.9, 0.1])[0]
    wait_time_category = random.choices(wait_time_categories, weights=[0.5, 0.3, 0.2])[0]

    data.append([
        client_company, division, region, service_type, service_cost,
        customer_name, customer_contact, customer_address,
        date_interview, invoice_date, close_date,
        *scores, average_score, nps_label,
        was_queue_long, was_staff_polite, used_mobile_banking, issue_resolved, wait_time_category
    ])

# Column names in English
columns = [
    'ClientCompany', 'Division', 'Region', 'ServiceType', 'ServiceCost',
    'CustomerName', 'CustomerContact', 'CustomerAddress',
    'DateInterviewed', 'InvoiceDate', 'WorkOrderCloseDate'
] + [f'Q{i+1}_Score' for i in range(n_questions)] + ['AverageScore', 'NPS_Label'] + thematic_flags

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('us_bank_feedback_dataset.csv', index=False, encoding='utf-8-sig')