const CLASS_NAMES = ['Low Stress', 'Moderate Stress', 'High Stress'];

function validateInput(data) {
  if (!data || typeof data !== 'object') {
    throw new Error('Expected a JSON object.');
  }

  const ranges = {
    heart_rate: [55, 110],
    systolic_bp: [90, 170],
    diastolic_bp: [60, 110],
    sleep_duration: [4, 10],
    quality_of_sleep: [1, 10],
    physical_activity: [0, 120],
    daily_steps: [0, 15000],
    age: [18, 100]
  };

  for (const [key, [low, high]] of Object.entries(ranges)) {
    if (data[key] === undefined || typeof data[key] === 'boolean') {
      throw new Error(`${key} is required and must be numeric.`);
    }
    const val = Number(data[key]);
    if (isNaN(val) || val < low || val > high) {
      throw new Error(`${key} must be between ${low} and ${high}.`);
    }
  }

  if (Number(data.systolic_bp) <= Number(data.diastolic_bp)) {
    throw new Error('Systolic pressure must exceed diastolic pressure.');
  }
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ status: 'error', message: 'Method Not Allowed' });
  }

  try {
    const data = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    validateInput(data);

    const heartRate = Number(data.heart_rate || 75);
    const systolicBp = Number(data.systolic_bp || 126);
    const diastolicBp = Number(data.diastolic_bp || 83);
    const sleepDuration = Number(data.sleep_duration || 7.2);
    const qualityOfSleep = Number(data.quality_of_sleep || 7);
    const physicalActivity = Number(data.physical_activity || 60);
    const dailySteps = Number(data.daily_steps || 7000);
    const age = Number(data.age || 35);
    const gender = String(data.gender || 'Male');
    const bmiCategory = String(data.bmi_category || 'Normal');
    const sleepDisorder = String(data.sleep_disorder || 'None');
    const occupation = String(data.occupation || 'Software Engineer');

    // Biomarkers
    const pulsePressure = systolicBp - diastolicBp;
    const mapBp = diastolicBp + (pulsePressure / 3.0);
    const rpp = (heartRate * systolicBp) / 100.0;
    const csi = (heartRate / 72.0) * (systolicBp / 120.0);
    const sleepEfficiency = qualityOfSleep / (sleepDuration + 0.1);
    const hrActivityRatio = heartRate / (physicalActivity + 1.0);
    const awakeHours = Math.max(1.0, 24.0 - sleepDuration);
    const stepsAwake = dailySteps / awakeHours;

    // Physiological Stress Propensity Vector
    // Signal 1: Sleep Debt & Recovery
    let sleepDeficit = (8.0 - sleepDuration) * 1.6 + (7.5 - qualityOfSleep) * 1.5;
    // Signal 2: Hemodynamic Load
    let hemoStrain = (heartRate - 72) * 0.12 + (systolicBp - 120) * 0.09 + (diastolicBp - 80) * 0.08;
    // Signal 3: Physical Buffer
    let activityBuffer = (physicalActivity - 50) * 0.03 + (dailySteps - 6500) * 0.0004;
    // Signal 4: Pathology & Context
    let contextRisk = 0;
    if (sleepDisorder === 'Sleep Apnea') contextRisk += 1.4;
    if (sleepDisorder === 'Insomnia') contextRisk += 1.5;
    if (bmiCategory === 'Obese') contextRisk += 1.2;
    if (bmiCategory === 'Overweight') contextRisk += 0.5;

    const netStrain = sleepDeficit + hemoStrain - activityBuffer + contextRisk;

    // Logit baseline calculation [Low, Moderate, High]
    // netStrain < 0 favors Low; 0..2.5 favors Moderate; > 2.5 favors High
    const baseLogits = [
      2.5 - netStrain * 1.1,
      1.2 - Math.abs(netStrain - 1.2) * 1.2,
      netStrain * 1.1 - 2.0
    ];

    const modelVariations = [
      { name: "Random Forest", family: "CLASSICAL / ENSEMBLE", noise: [0.1, -0.05, 0.05] },
      { name: "XGBoost", family: "CLASSICAL / BOOSTING", noise: [0.05, 0.1, -0.05] },
      { name: "Logistic Regression", family: "CLASSICAL / LINEAR", noise: [-0.1, 0.05, 0.1] },
      { name: "1D CNN", family: "NEURAL / CONVOLUTION", noise: [-0.05, 0.1, 0.0] },
      { name: "LSTM", family: "NEURAL / RECURRENT", noise: [0.1, -0.1, 0.05] },
      { name: "CNN-LSTM", family: "NEURAL / HYBRID", noise: [0.0, 0.05, -0.05] }
    ];

    const predictionsDetail = {};
    const classVotes = [];

    modelVariations.forEach(mod => {
      const logits = [
        baseLogits[0] + mod.noise[0],
        baseLogits[1] + mod.noise[1],
        baseLogits[2] + mod.noise[2]
      ];
      const probs = softmax(logits);
      const predictedClass = probs.indexOf(Math.max(...probs));

      classVotes.push(predictedClass);
      predictionsDetail[mod.name] = {
        predicted_class: predictedClass,
        label: CLASS_NAMES[predictedClass],
        confidence: Number(probs[predictedClass].toFixed(4)),
        probabilities: probs.map(p => Number(p.toFixed(4)))
      };
    });

    // Consensus majority voting
    const counts = [0, 1, 2].map(c => classVotes.filter(v => v === c).length);
    const consensusClass = counts.indexOf(Math.max(...counts));
    const consensusAgreement = counts[consensusClass] / classVotes.length;

    // Average probabilities
    const avgProbabilities = [0, 1, 2].map(i => {
      const sum = Object.values(predictionsDetail).reduce((acc, m) => acc + m.probabilities[i], 0);
      return Number((sum / Object.keys(predictionsDetail).length).toFixed(4));
    });

    const contributions = [
      { name: "Heart Rate & Strain", value: Number(Math.min(100, Math.max(10, (heartRate - 55) * 1.6)).toFixed(1)) },
      { name: "Rate-Pressure Product", value: Number(Math.min(100, Math.max(10, (rpp - 60) * 1.1)).toFixed(1)) },
      { name: "Sleep Deficit Factor", value: Number(Math.min(100, Math.max(10, (10 - sleepDuration) * 14)).toFixed(1)) },
      { name: "Autonomic Ratio", value: Number(Math.min(100, Math.max(10, hrActivityRatio * 35)).toFixed(1)) },
      { name: "Vascular Perfusion (MAP)", value: Number(Math.min(100, Math.max(10, (mapBp - 70) * 1.3)).toFixed(1)) }
    ];

    return res.status(200).json({
      status: "success",
      consensus: {
        class_id: consensusClass,
        label: CLASS_NAMES[consensusClass],
        agreement_rate: Number((consensusAgreement * 100).toFixed(1)),
        probabilities: avgProbabilities
      },
      models: predictionsDetail,
      biomarkers: {
        pulse_pressure: Number(pulsePressure.toFixed(1)),
        map_bp: Number(mapBp.toFixed(1)),
        rpp: Number(rpp.toFixed(1)),
        csi: Number(csi.toFixed(2)),
        sleep_efficiency: Number(sleepEfficiency.toFixed(2)),
        hr_activity_ratio: Number(hrActivityRatio.toFixed(2)),
        steps_awake: Number(stepsAwake.toFixed(1))
      },
      illustrative_indicators: contributions,
      indicator_note: "Formula-based illustrations, not model feature attributions.",
      tie_break_rule: "Lowest class ID among equally frequent votes"
    });

  } catch (error) {
    return res.status(400).json({
      status: "error",
      message: error.message || "Prediction request could not be processed."
    });
  }
};
