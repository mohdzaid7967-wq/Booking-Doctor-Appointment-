import pandas as pd
from langchain_core.tools import tool
from datetime import datetime
import random

FILE_PATH = "data/doctor_availability.csv"


def normalize_date(date_str: str) -> str:
    """Normalize various date formats to DD-MM-YYYY."""
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %m %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return date_str.strip()


def normalize_time(time_str: str) -> str:
    """Normalize time to HH:MM 24h format."""
    time_str = time_str.strip()
    for fmt in ("%H:%M", "%I:%M%p", "%I:%M %p", "%I%p", "%Ipm", "%Iam"):
        try:
            return datetime.strptime(time_str.upper(), fmt.upper()).strftime("%H:%M")
        except ValueError:
            continue
    # handle plain '14' or '2pm'
    try:
        if time_str.lower().endswith('pm') or time_str.lower().endswith('am'):
            return datetime.strptime(time_str, "%I%p").strftime("%H:%M")
    except Exception:
        pass
    return time_str

def clean_doctor_name(name: str) -> str:
    """Removes 'Dr.' prefix and cleans up the string for robust matching."""
    name = name.lower().strip()
    name = name.replace("dr.", "").replace("dr ", "").strip()
    return name

def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception:
        return None

    df["doctor_name"] = df["doctor_name"].str.strip().str.lower()
    df["specialization"] = df["specialization"].str.strip().str.lower()
    df["date_only"] = df["date_slot"].str.split(" ").str[0]
    df["is_available"] = df["is_available"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)

    return df


@tool
def check_availability_by_doctor(doctor_name: str, desired_date: str) -> str:
    """
    Check available time slots for a specific doctor on a given date.
    Args:
        doctor_name: Full or partial name of the doctor (e.g. 'john doe')
        desired_date: Date in DD-MM-YYYY format (e.g. '09-08-2026')
    """
    df = load_data()
    if df is None:
        return "Data file not found."

    doctor_name = clean_doctor_name(doctor_name)
    desired_date = normalize_date(desired_date)

    result = df[
        (df["doctor_name"].str.contains(doctor_name, case=False, na=False)) &
        (df["date_only"] == desired_date)
    ]

    if result.empty:
        return f"No records found for Dr. {doctor_name.title()} on {desired_date}. The date may not exist in the schedule."

    available = result[result["is_available"] == True]

    if available.empty:
        return f"Dr. {doctor_name.title()} has no free slots on {desired_date}. All slots are booked."

    slots = available["date_slot"].tolist()
    slots_str = "\n".join(f"  - {s}" for s in slots[:10])
    return f"Available slots for Dr. {doctor_name.title()} on {desired_date}:\n{slots_str}"


@tool
def check_availability_by_specialization(specialization: str, desired_date: str) -> str:
    """
    Check available doctors and slots for a specialization on a given date.
    Args:
        specialization: Type of doctor (e.g. 'general_dentist', 'cosmetic_dentist')
        desired_date: Date in DD-MM-YYYY format (e.g. '09-08-2026')
    """
    df = load_data()
    if df is None:
        return "Data file not found."

    specialization = specialization.strip().lower()
    desired_date = normalize_date(desired_date)

    result = df[
        (df["specialization"].str.contains(specialization, case=False, na=False)) &
        (df["date_only"] == desired_date)
    ]

    if result.empty:
        return f"No doctors found for specialization '{specialization}' on {desired_date}."

    available = result[result["is_available"] == True]

    if available.empty:
        return f"No free slots for any {specialization} on {desired_date}. All slots are fully booked."

    lines = []
    for _, row in available.head(10).iterrows():
        lines.append(f"  - Dr. {row['doctor_name'].title()} at {row['date_slot']}")

    return f"Available {specialization} slots on {desired_date}:\n" + "\n".join(lines)


@tool
def set_appointment(doctor_name: str, desired_date: str, desired_time: str, patient_id: int = 0) -> str:
    """
    Book an appointment with a doctor. If the user doesn't have a patient_id, pass 0 to automatically generate one.
    Args:
        doctor_name: Full name of the doctor (e.g. 'john doe')
        desired_date: Date in DD-MM-YYYY format (e.g. '09-08-2026')
        desired_time: Time in HH:MM 24h format (e.g. '14:00')
        patient_id: Integer patient ID. Pass 0 to generate a new ID for new patients.
    """
    raw_df = pd.read_csv(FILE_PATH)
    df = load_data()
    if df is None:
        return "Data file not found."

    doctor_name = clean_doctor_name(doctor_name)
    desired_date = normalize_date(desired_date)
    desired_time = normalize_time(desired_time)
    slot = f"{desired_date} {desired_time}"

    condition = (
        (df["doctor_name"].str.contains(doctor_name, case=False, na=False)) &
        (df["date_slot"] == slot)
    )

    matched = df[condition]
    if matched.empty:
        return f"No slot found for Dr. {doctor_name.title()} at {slot}. Please verify the doctor name, date and time."

    if not matched["is_available"].values[0]:
        return f"The slot {slot} for Dr. {doctor_name.title()} is already booked. Please choose another time."

    # Generate a unique patient_id if none was provided
    if not patient_id or patient_id == 0:
        patient_id = random.randint(1000000, 9999999)
        while float(patient_id) in raw_df["patient_to_attend"].values:
            patient_id = random.randint(1000000, 9999999)

    raw_condition = (
        (raw_df["doctor_name"].str.strip().str.lower().str.contains(doctor_name, case=False, na=False)) &
        (raw_df["date_slot"] == slot)
    )
    raw_df.loc[raw_condition, "is_available"] = False
    raw_df.loc[raw_condition, "patient_to_attend"] = float(patient_id)
    raw_df.to_csv(FILE_PATH, index=False)

    return f"Appointment successfully booked with Dr. {doctor_name.title()} on {slot}. YOUR NEW PATIENT ID IS {patient_id}. Please inform the user to save this ID!"


@tool
def cancel_appointment(doctor_name: str, desired_date: str, desired_time: str, patient_id: int) -> str:
    """
    Cancel an existing appointment.
    Args:
        doctor_name: Full name of the doctor
        desired_date: Date in DD-MM-YYYY format
        desired_time: Time in HH:MM 24h format
        patient_id: Required Integer patient ID 
    """
    if not patient_id or patient_id == 0:
        return "A valid patient_id is required to cancel an appointment."
        
    raw_df = pd.read_csv(FILE_PATH)
    df = load_data()
    if df is None:
        return "Data file not found."

    doctor_name = clean_doctor_name(doctor_name)
    desired_date = normalize_date(desired_date)
    desired_time = normalize_time(desired_time)
    slot = f"{desired_date} {desired_time}"

    condition = (
        (df["doctor_name"].str.contains(doctor_name, case=False, na=False)) &
        (df["date_slot"] == slot) &
        (df["patient_to_attend"] == float(patient_id))
    )

    if df[condition].empty:
        return f"No appointment found for patient {patient_id} with Dr. {doctor_name.title()} at {slot}."

    raw_condition = (
        (raw_df["doctor_name"].str.strip().str.lower().str.contains(doctor_name, case=False, na=False)) &
        (raw_df["date_slot"] == slot) &
        (raw_df["patient_to_attend"] == float(patient_id))
    )
    raw_df.loc[raw_condition, "is_available"] = True
    raw_df.loc[raw_condition, "patient_to_attend"] = None
    raw_df.to_csv(FILE_PATH, index=False)

    return f"Appointment cancelled successfully. Dr. {doctor_name.title()} on {slot} for patient {patient_id}."


@tool
def reschedule_appointment(doctor_name: str, old_date: str, old_time: str, new_date: str, new_time: str, patient_id: int) -> str:
    """
    Reschedule an existing appointment to a new date and time.
    Args:
        doctor_name: Full name of the doctor
        old_date: Current appointment date in DD-MM-YYYY format
        old_time: Current appointment time in HH:MM format
        new_date: New appointment date in DD-MM-YYYY format
        new_time: New appointment time in HH:MM format
        patient_id: Required Integer patient ID
    """
    if not patient_id or patient_id == 0:
        return "A valid patient_id is required to reschedule an appointment."
        
    cancel_msg = cancel_appointment.invoke({
        "doctor_name": doctor_name,
        "desired_date": old_date,
        "desired_time": old_time,
        "patient_id": patient_id
    })

    if "No appointment found" in cancel_msg or "not found" in cancel_msg.lower():
        return f"Could not reschedule: {cancel_msg}"

    book_msg = set_appointment.invoke({
        "doctor_name": doctor_name,
        "desired_date": new_date,
        "desired_time": new_time,
        "patient_id": patient_id
    })

    return f"Rescheduled successfully!\nOld: {cancel_msg}\nNew: {book_msg}"