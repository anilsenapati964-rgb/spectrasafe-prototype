def route_for(decision):
    return {"decision": decision, "action": "QUARANTINE" if decision == "FLAG" else "MAIN LINE",
            "actuator": "AIR-JET / SERVO SIMULATED — ACTIVATED" if decision == "FLAG" else "NO REJECTION — CONTINUE ON MAIN LINE"}
