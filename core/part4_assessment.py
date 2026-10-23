"""
Part 4: Crop Health Assessment (Features + Environmental Data)
- Cross-analyzes visual disease features with continuous IoT sensor telemetry
- Identifies environmental abiotic stress triggers
- Computes consolidated Crop Health Score (0 - 100)
- Prescribes actionable chemical & cultural management dosages
"""

AGRONOMIC_PRESCRIPTIONS = {
    "Tomato - Healthy": {
        "Diagnosis": "Plant physiology optimal; zero fungal/bacterial necrosis.",
        "Abiotic Trigger": "Balanced environmental microclimate.",
        "Chemical Treatment": "No chemical intervention needed.",
        "Dosage": "N/A",
        "Cultural Practice": "Maintain scheduled drip irrigation; conduct bi-weekly soil NPK testing."
    },
    "Tomato - Early Blight": {
        "Diagnosis": "Foliar fungal infection caused by Alternaria solani.",
        "Abiotic Trigger": "Elevated relative humidity (>80%) creating microclimate conducive to spore germination.",
        "Chemical Treatment": "Chlorothalonil 75 WP or Copper Oxychloride 50 WP",
        "Dosage": "2.0 g/L of water (Chlorothalonil) or 3.0 g/L (Copper Oxychloride)",
        "Cultural Practice": "Avoid overhead sprinkler wetting; prune lower infected foliage; mulch base to stop splash."
    },
    "Potato - Late Blight": {
        "Diagnosis": "Aggressive oomycete pathogen: Phytophthora infestans.",
        "Abiotic Trigger": "Excessive soil waterlogging (>70%) combined with cool, humid canopy microclimate.",
        "Chemical Treatment": "Metalaxyl-M 4% + Mancozeb 64% WP (Ridomil Gold) or Cymoxanil 8% + Mancozeb 64% WP",
        "Dosage": "2.5 g/L of water immediately via full foliar coverage",
        "Cultural Practice": "Improve furrow drainage immediately; destroy severely infected cull piles."
    },
    "Corn - Common Rust": {
        "Diagnosis": "Basidiomycete fungal pathogen: Puccinia sorghi.",
        "Abiotic Trigger": "Warm ambient temperature (>30°C) combined with prolonged morning dew and high humidity.",
        "Chemical Treatment": "Azoxystrobin 23 SC or Pyraclostrobin 20% WG",
        "Dosage": "1.0 mL/L of water (Azoxystrobin) or 1.0 g/L (Pyraclostrobin)",
        "Cultural Practice": "Plant resistant hybrids; ensure wider row spacing for enhanced canopy ventilation."
    }
}

def assess_crop_health(ai_result, telemetry):
    """
    Multimodal Crop Health Evaluator:
    Combines AI visual disease diagnosis with IoT sensor telemetry.
    """
    moisture = float(telemetry.get("soil_moisture", 45.0))
    temp = float(telemetry.get("temperature", 25.0))
    humidity = float(telemetry.get("humidity", 65.0))
    ph = float(telemetry.get("soil_ph", 6.5))

    detected_class = ai_result["detected_class"]
    severity_score = float(ai_result["severity_score"])

    # Calculate Abiotic Stress Penalties & Triggers
    abiotic_triggers = []
    abiotic_penalty = 0.0

    if humidity > 80.0:
        abiotic_triggers.append(f"High Relative Humidity ({humidity:.1f}%) — Conducive fungal microclimate")
        abiotic_penalty += 35.0
    elif humidity < 35.0:
        abiotic_triggers.append(f"Low Ambient Humidity ({humidity:.1f}%) — Transpiration stress")
        abiotic_penalty += 15.0

    if moisture > 70.0:
        abiotic_triggers.append(f"Excessive Soil Moisture ({moisture:.1f}%) — Root zone waterlogging risk")
        abiotic_penalty += 25.0
    elif moisture < 30.0:
        abiotic_triggers.append(f"Severe Soil Moisture Deficit ({moisture:.1f}%) — Moisture drought stress")
        abiotic_penalty += 25.0

    if temp > 33.0:
        abiotic_triggers.append(f"Elevated Ambient Heat ({temp:.1f}°C) — Thermal heat shock")
        abiotic_penalty += 20.0
    elif temp < 16.0:
        abiotic_triggers.append(f"Low Ambient Temperature ({temp:.1f}°C) — Chilling inhibition")
        abiotic_penalty += 15.0

    if ph < 5.8:
        abiotic_triggers.append(f"Acidic Soil (pH {ph:.2f}) — Inhibits phosphorus & magnesium uptake")
        abiotic_penalty += 10.0
    elif ph > 7.5:
        abiotic_triggers.append(f"Alkaline Soil (pH {ph:.2f}) — Micronutrient availability restriction")
        abiotic_penalty += 10.0

    if not abiotic_triggers:
        abiotic_triggers.append("Environmental parameters are within optimal agronomic thresholds.")

    # Multimodal Total Stress Index (0 - 100)
    # Visual Disease Severity (60% weight) + Abiotic Sensor Stress (40% weight)
    total_stress_index = min(100.0, (severity_score * 0.6) + (abiotic_penalty * 0.4))
    health_score = round(max(0.0, 100.0 - total_stress_index), 1)

    if health_score >= 80.0:
        health_status = "OPTIMAL / LOW RISK"
        status_color = "green"
    elif health_score >= 50.0:
        health_status = "MODERATE STRESS / MONITOR CLOSELY"
        status_color = "orange"
    else:
        health_status = "CRITICAL / IMMEDIATE INTERVENTION REQUIRED"
        status_color = "red"

    prescription = AGRONOMIC_PRESCRIPTIONS.get(detected_class, AGRONOMIC_PRESCRIPTIONS["Tomato - Healthy"])

    return {
        "health_score": health_score,
        "health_status": health_status,
        "status_color": status_color,
        "abiotic_triggers": abiotic_triggers,
        "prescription": prescription
    }
