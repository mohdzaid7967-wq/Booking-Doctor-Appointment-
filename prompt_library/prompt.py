# Prompt constants are now maintained directly in agent.py as module-level strings.
# This file is kept for backward compatibility.

members_dict = {
    'information_node': 'Specialized agent to provide information related to availability of doctors or any FAQs related to hospital.',
    'booking_node': 'Specialized agent to only book, cancel or reschedule appointment'
}

options = list(members_dict.keys()) + ["FINISH"]