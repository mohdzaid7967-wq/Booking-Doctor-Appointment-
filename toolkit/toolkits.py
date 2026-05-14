import pandas as pd
from langchain_core.tools import tool

FILE_PATH = "data/doctor_availability.csv"


# ✅ Helper function
def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception:
        return None

    df["doctor_name"] = df["doctor_name"].str.strip().str.lower()
    df["specialization"] = df["specialization"].str.strip().str.lower()
    df["date_only"] = df["date_slot"].str.split(" ").str[0]
    df["is_available"] = df["is_available"].astype(bool)

    return df


@tool
def check_availability_by_doctor(doctor_name: str, desired_date: str):
    """Check available slots for a doctor on a specific date."""
    
    df = load_data()
    if df is None:
        return "❌ Data file not found"

    doctor_name = doctor_name.strip().lower()

    result = df[
        (df["doctor_name"] == doctor_name) &
        (df["date_only"] == desired_date)
    ]

    if result.empty:
        return f"No availability found for {doctor_name} on {desired_date}"

    available = result[result["is_available"] == True]

    if available.empty:
        return f"No free slots available for {doctor_name} on {desired_date}"

    return available[["date_slot"]].to_string(index=False)


@tool
def check_availability_by_specialization(specialization: str, desired_date: str):
    """Check available slots for a specialization on a specific date."""
    
    df = load_data()
    if df is None:
        return "❌ Data file not found"

    specialization = specialization.strip().lower()

    result = df[
        (df["specialization"] == specialization) &
        (df["date_only"] == desired_date)
    ]

    if result.empty:
        return f"No doctors found for {specialization} on {desired_date}"

    available = result[result["is_available"] == True]

    if available.empty:
        return f"No free slots for {specialization} on {desired_date}"

    return available[["doctor_name", "date_slot"]].to_string(index=False)


@tool
def set_appointment(doctor_name: str, desired_date: str, desired_time: str, patient_id: int):
    """Book an appointment with a doctor."""
    
    df = load_data()
    if df is None:
        return "❌ Data file not found"

    doctor_name = doctor_name.strip().lower()
    slot = f"{desired_date} {desired_time}"

    condition = (
        (df["doctor_name"] == doctor_name) &
        (df["date_slot"] == slot)
    )

    if not df[condition].empty and df.loc[condition, "is_available"].values[0]:
        df.loc[condition, "is_available"] = False
        df.loc[condition, "patient_id"] = patient_id

        df.to_csv(FILE_PATH, index=False)
        return f"✅ Appointment booked with {doctor_name} on {slot}"

    return f"❌ Slot not available for {doctor_name} at {slot}"


@tool
def cancel_appointment(doctor_name: str, desired_date: str, patient_id: int):
    """Cancel an existing appointment."""
    
    df = load_data()
    if df is None:
        return "❌ Data file not found"

    doctor_name = doctor_name.strip().lower()

    condition = (
        (df["doctor_name"] == doctor_name) &
        (df["date_only"] == desired_date) &
        (df["patient_id"] == patient_id)
    )

    if df[condition].empty:
        return "❌ No appointment found to cancel"

    df.loc[condition, "is_available"] = True
    df.loc[condition, "patient_id"] = None

    df.to_csv(FILE_PATH, index=False)

    return f"✅ Appointment cancelled for {doctor_name} on {desired_date}"


@tool
def reschedule_appointment(doctor_name: str, old_date: str, new_date: str, new_time: str, patient_id: int):
    """Reschedule an appointment."""
    
    cancel_msg = cancel_appointment.invoke({
        "doctor_name": doctor_name,
        "desired_date": old_date,
        "patient_id": patient_id
    })

    book_msg = set_appointment.invoke({
        "doctor_name": doctor_name,
        "desired_date": new_date,
        "desired_time": new_time,
        "patient_id": patient_id
    })

    return f"{cancel_msg}\n{book_msg}"