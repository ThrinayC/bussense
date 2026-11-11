/**
 * BusSense — Road Repair Cost Estimation & CSR Match Engine
 * ==========================================================
 * Project-defined transparent civil engineering cost model for road defects.
 * Computes defensible estimates derived from actual stereo metrology
 * (depth_cm, area_m2, severity, road_condition).
 * 
 * DISCLAIMER:
 * Engineering estimate derived from measured defect geometry and configurable
 * cold-mix polymer asphalt repair assumptions. Not an official municipal quotation.
 */

export const CSR_REPAIR_COST_CONFIG = {
  // Asphalt Material Rates
  material_base_inr: 800.0,            // Base mobilization for hot/cold polymer mix
  material_rate_per_m3_inr: 15000.0,   // Rate per cubic meter of compacted polymer cold-mix
  tack_coat_per_m2_inr: 1200.0,        // Bitumen emulsion tack coat per m²
  
  // Labour & Compaction Crew
  labour_base_inr: 1400.0,             // Minimum skilled 2-person crew deployment
  labour_rate_per_m2_inr: 1100.0,      // Surface preparation, squaring, edge compaction per m²
  
  // Heavy Equipment Allowance
  equipment_base_inr: 1500.0,          // Plate compactor, pavement saw, asphalt vibratory roller
  
  // Traffic Management on Active Transit Corridor
  traffic_management_by_severity: {
    CRITICAL: 1600.0,                  // Full lane channeling, reflective cones, flagger
    HIGH: 1200.0,                      // Lane tapering, caution signage
    MEDIUM: 800.0,                     // Portable barricades
    LOW: 500.0,                        // Standard shoulder cones
  },
  
  // Contingency & Site Quality Assurance
  contingency_percentage: 0.10,        // 10% contingency on sum of physical costs
  
  // Crack Sealing Unit Rates (Linear crack repair)
  crack_base_sealant_inr: 900.0,
  crack_rate_per_linear_meter_inr: 450.0,
  crack_per_branch_allowance_inr: 300.0,
  
  disclaimer: "Engineering estimate based on BusSense stereo metrology (depth/area) and standard municipal cold-mix asphalt schedules. Not an official municipal tender quotation."
};

/**
 * Calculates a transparent, defensible repair cost breakdown for any road defect.
 * 
 * @param {object} defect - Road defect event (POTHOLE or CRACK)
 * @returns {object} Breakdown of all cost components and total in INR
 */
export function calculateRepairCost(defect) {
  if (!defect) return null;

  const type = (defect.event_type || defect.defect_type || '').toUpperCase();
  const isPothole = type === 'POTHOLE';
  const severity = (defect.severity || 'MEDIUM').toUpperCase();

  if (isPothole) {
    const area = Math.max(Number(defect.area_m2) || 0.12, 0.04);
    const depthCm = Math.max(Number(defect.depth_cm) || 2.5, 1.0);
    const depthM = depthCm / 100.0;
    const volumeM3 = area * depthM;

    // 1. Material Cost: Volume * asphalt density factor + tack coat
    const materialCost = Math.round(
      CSR_REPAIR_COST_CONFIG.material_base_inr +
      (volumeM3 * CSR_REPAIR_COST_CONFIG.material_rate_per_m3_inr) +
      (area * CSR_REPAIR_COST_CONFIG.tack_coat_per_m2_inr)
    );

    // 2. Labour Cost: Base + area compaction
    const labourCost = Math.round(
      CSR_REPAIR_COST_CONFIG.labour_base_inr +
      (area * CSR_REPAIR_COST_CONFIG.labour_rate_per_m2_inr)
    );

    // 3. Equipment Allowance
    const equipmentCost = Math.round(CSR_REPAIR_COST_CONFIG.equipment_base_inr);

    // 4. Traffic Management
    const trafficCost = Math.round(
      CSR_REPAIR_COST_CONFIG.traffic_management_by_severity[severity] || 1000.0
    );

    // 5. Subtotal & Contingency
    const subtotal = materialCost + labourCost + equipmentCost + trafficCost;
    const contingency = Math.round(subtotal * CSR_REPAIR_COST_CONFIG.contingency_percentage);
    
    // Round total to nearest ₹50
    const rawTotal = subtotal + contingency;
    const totalCost = Math.round(rawTotal / 50.0) * 50;

    return {
      defect_type: 'POTHOLE',
      area_m2: area,
      depth_cm: depthCm,
      volume_m3: Number(volumeM3.toFixed(4)),
      severity,
      material_cost: materialCost,
      labour_cost: labourCost,
      equipment_cost: equipmentCost,
      traffic_management: trafficCost,
      contingency: contingency,
      total_cost: totalCost,
      disclaimer: CSR_REPAIR_COST_CONFIG.disclaimer
    };
  } else {
    // Crack network sealing
    const lengthM = Math.max(Number(defect.length_m) || 1.8, 0.5);
    const branches = Math.max(Number(defect.branches) || 1, 0);

    const materialCost = Math.round(
      CSR_REPAIR_COST_CONFIG.crack_base_sealant_inr +
      (lengthM * CSR_REPAIR_COST_CONFIG.crack_rate_per_linear_meter_inr) +
      (branches * CSR_REPAIR_COST_CONFIG.crack_per_branch_allowance_inr)
    );

    const labourCost = Math.round(CSR_REPAIR_COST_CONFIG.labour_base_inr + (lengthM * 350.0));
    const equipmentCost = Math.round(CSR_REPAIR_COST_CONFIG.equipment_base_inr * 0.7);
    const trafficCost = Math.round(CSR_REPAIR_COST_CONFIG.traffic_management_by_severity[severity] || 800.0);

    const subtotal = materialCost + labourCost + equipmentCost + trafficCost;
    const contingency = Math.round(subtotal * CSR_REPAIR_COST_CONFIG.contingency_percentage);
    const totalCost = Math.round((subtotal + contingency) / 50.0) * 50;

    return {
      defect_type: 'CRACK',
      length_m: lengthM,
      branches: branches,
      severity,
      material_cost: materialCost,
      labour_cost: labourCost,
      equipment_cost: equipmentCost,
      traffic_management: trafficCost,
      contingency: contingency,
      total_cost: totalCost,
      disclaimer: CSR_REPAIR_COST_CONFIG.disclaimer
    };
  }
}

/**
 * Formats a monetary number into Indian Currency format (e.g. ₹5,850 or ₹50,00,000)
 */
export function formatINR(val) {
  if (val == null || isNaN(val)) return '₹—';
  const num = Math.round(Number(val));
  const s = num.toString();
  if (s.length <= 3) return '₹' + s;
  
  const last3 = s.slice(-3);
  let other = s.slice(0, -3);
  let res = '';
  while (other.length > 2) {
    res = ',' + other.slice(-2) + res;
    other = other.slice(0, -2);
  }
  return '₹' + other + res + ',' + last3;
}

/**
 * Transparent CSR partner matching rules.
 * Matches road defects to potential CSR sponsors based on:
 * 1. Geographic compatibility (Tiruchirappalli / Tamil Nadu)
 * 2. Focus area relevance (Road Safety, Pavement, Transit Corridors)
 * 3. Budget sufficiency (Available allocation >= Repair Estimate)
 * 
 * @param {object} defect - The pothole/defect
 * @param {Array} partners - List of CSR partner profiles
 * @returns {Array} List of partners with matching tags and compatibility rank
 */
export function matchCsrPartners(defect, partners = []) {
  const estimate = calculateRepairCost(defect);
  const cost = estimate?.total_cost || 5000;
  const severity = (defect.severity || 'MEDIUM').toUpperCase();

  return partners.map(p => {
    const reasons = [];
    let matchScore = 0; // 0 to 100

    // 1. Geography match
    const geo = (p.geography || '').toLowerCase();
    if (geo.includes('trichy') || geo.includes('tiruchirappalli')) {
      reasons.push('Direct Tiruchirappalli Corridor Focus');
      matchScore += 35;
    } else if (geo.includes('tamil nadu')) {
      reasons.push('Statewide Tamil Nadu Road Program');
      matchScore += 25;
    }

    // 2. Focus match
    const focusList = Array.isArray(p.focus_areas) ? p.focus_areas : [];
    const focusText = focusList.join(' ').toLowerCase();
    if (focusText.includes('pothole') || focusText.includes('safety') || focusText.includes('transit')) {
      reasons.push('Road Safety & Transit Priority');
      matchScore += 35;
    } else if (focusText.includes('infrastructure') || focusText.includes('pavement')) {
      reasons.push('Pavement & Civil Infrastructure Focus');
      matchScore += 30;
    }

    // 3. Budget sufficiency
    const available = Number(p.available_budget_inr) || 0;
    if (available >= cost * 10) {
      reasons.push(`Ample CSR Budget (Avail: ${formatINR(available)})`);
      matchScore += 30;
    } else if (available >= cost) {
      reasons.push(`Sufficient CSR Allocation (Avail: ${formatINR(available)})`);
      matchScore += 20;
    } else {
      reasons.push(`Budget Constrained (Deficit: ${formatINR(cost - available)})`);
      matchScore -= 20;
    }

    // 4. Severity urgency bonus
    if (severity === 'CRITICAL' && matchScore > 50) {
      matchScore = Math.min(100, matchScore + 10);
    }

    return {
      ...p,
      match_score: Math.max(10, Math.min(98, matchScore)),
      match_reasons: reasons,
      is_recommended: matchScore >= 70 && available >= cost,
      can_fund: available >= cost
    };
  }).sort((a, b) => b.match_score - a.match_score);
}
