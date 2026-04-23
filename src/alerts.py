def generate_alert(rul):

    if rul < 10:
        return "🔴 CRITICAL"
    elif rul < 20:
        return "🟠 HIGH"
    elif rul < 50:
        return "🟡 WARNING"
    else:
        return "🟢 HEALTHY"