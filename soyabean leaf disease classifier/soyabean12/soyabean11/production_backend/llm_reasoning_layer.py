"""
Dynamic LLM Reasoning Layer - Intelligence Upgrade
Generates context-aware agricultural advice based on CNN predictions with variant knowledge system
"""
import logging
import json
from typing import Dict, Any, Optional
import re
import random

logger = logging.getLogger(__name__)

# Disease Knowledge Variants Database (5 variants per disease)
DISEASE_KNOWLEDGE_VARIANTS = {
    "Anthracnose": [
        {
            "variant_focus": "Weather-Driven Outbreak",
            "insight": "Anthracnose (caused by *Colletotrichum truncatum*) thrives during prolonged warm, wet weather (75-85°F with extended leaf wetness). Infection begins when fungal spores land on leaf surfaces during rain splash or overhead irrigation. The pathogen penetrates plant tissue through stomata or wounded areas, establishing itself within 48-72 hours under optimal humidity (>90%). Reproductive stage (R3-R5) plants are most vulnerable as dense canopy traps moisture.",
            "visual_identification": "**Early Stage:** Small, irregular brown lesions with dark borders appear on upper leaf surfaces. **Mid Stage:** Lesions expand to 1-2cm with characteristic tan centers and purple-black margins. Veins within lesions turn dark brown. **Late Stage:** Severe defoliation as lesions coalesce. Stems show elongated reddish-brown cankers. Pods develop sunken black spots with pink spore masses during humid mornings. Differs from Brown Spot by having darker, more angular lesions with distinct red-purple halos.",
            "risk_impact": "**Yield Loss:** 10-20% in mild cases; 40-60% in severe epidemics if pods are infected. **Quality Impact:** Infected seeds show shriveled appearance, reduced germination (30-50% loss), and lower oil content. **Field-Scale:** Pathogen overwinters in crop residue, creating inoculum reservoirs for 2-3 years. Adjacent fields face 15-25% higher infection risk during wet seasons.",
            "control_strategy": "**Chemical:** Apply QoI fungicides (strobilurin class) at R3 growth stage preventively. Follow with DMI fungicides (triazole class) 14 days later if weather remains humid. Rotate fungicide classes to prevent resistance. **Cultural:** Plant resistant varieties (check state extension lists). Ensure 30-inch row spacing for air circulation. Avoid irrigation during evening hours. **Mechanical:** Deep plow infected residue immediately post-harvest.",
            "monitoring_prevention": "**Next 3-7 Days:** Scout fields every 2 days, focusing on lower canopy where humidity is highest. Use a 10x hand lens to detect early lesions (2-3mm). Track daily leaf wetness hours using weather stations. **Next 14 Days:** If rain is forecasted (>0.5 inches), apply preventive fungicide within 24 hours before event. **Next Season:** Plant certified disease-free seeds. Rotate with non-legume crops (corn, wheat) for 2 years. Remove volunteer soybean plants that harbor spores."
        },
        {
            "variant_focus": "Early vs Late Infection Timeline",
            "insight": "*Colletotrichum truncatum* causes Anthracnose through wind-dispersed ascospores from infected crop debris. Infection timing determines impact severity: early vegetative infections (V3-V5) cause localized leaf damage with minimal yield loss, while reproductive-stage infections (R4-R6) directly attack pods and stems, causing 30-50% yield reduction. The fungus requires 6-12 hours of continuous leaf wetness at 72-82°F for successful colonization. Late-season infections coincide with pod fill, when plants have reduced defense capacity.",
            "visual_identification": "**Vegetative Stage:** Leaves show small (3-5mm) circular spots with gray-tan centers and dark brown borders. Lower leaves affected first. **Reproductive Stage:** Lesions enlarge rapidly to 1-3cm, often merging to cover entire leaflets. Premature yellowing and leaf drop occur within 5-7 days. **Pod/Stem Symptoms:** Black, sunken lesions on pods appear glossy with pinkish spore masses in wet conditions. Stem cankers appear as elongated dark streaks that girdle the plant. Unlike Frogeye Leaf Spot, Anthracnose lesions lack concentric ring patterns.",
            "risk_impact": "**Early Infection (V-Stage):** 5-15% yield loss, primarily through reduced photosynthetic capacity. Seeds remain largely unaffected. **Late Infection (R-Stage):** 40-70% loss due to premature plant death, pod infection, and poor seed fill. **Market Impact:** Infected seeds downgraded from Grade 1 to Grade 3, reducing value by $2-4/bushel. Field-wide infection forces early harvest, increasing green seed content and dockage penalties.",
            "control_strategy": "**Timing-Based Application:** If scouting detects lesions before R3, apply single preventive spray of azoxystrobin (11 fl oz/acre). For R3+ detections, use tank mix of trifloxystrobin + prothioconazole (6 fl oz/acre) immediately, repeat 14 days later. **Resistance Management:** Alternate FRAC groups (FRAC 3 → FRAC 11 → FRAC 7) across season. Never use same class twice consecutively. **Cultural Practices:** Use early-maturing varieties in high-pressure zones to escape late-season infection. Maintain field drainage to reduce surface water pooling.",
            "monitoring_prevention": "**Immediate (Days 1-5):** Flag all plants with visible lesions using colored tape for tracking expansion rate. Collect 20 leaflets per 5 acres for lab confirmation ($25/sample at extension offices). **Week 2-3:** Monitor weather closely—infection risk peaks when 3+ consecutive days have >85% humidity and temps 75-85°F. Apply protectant fungicide if risk threshold met. **Long-Term Prevention:** Submit soil samples post-harvest to test for *Colletotrichum* propagule counts. Populations >500 CFU/g indicate mandatory 2-year rotation away from soybeans."
        },
        {
            "variant_focus": "Yield Risk vs Quality Loss Assessment",
            "insight": "Anthracnose, caused by *Colletotrichum truncatum*, presents dual threats: immediate yield loss through defoliation and long-term quality degradation via seed infection. The fungus produces both sexual (ascospores) and asexual (conidia) spores, enabling rapid spread during warm (77-82°F), wet conditions. Spores germinate within 4-8 hours of leaf wetting, forming appressoria that penetrate plant cuticles. Early vegetative infections reduce photosynthetic capacity by 15-30%, while pod infections directly compromise seed viability. Plants at R5-R6 (seed fill) stages experience most severe impacts.",
            "visual_identification": "**Leaf Symptoms:** Begin as small, dark brown to black spots (2-4mm) on upper leaf surfaces. Progress to larger (5-15mm) irregular lesions with tan-gray centers and distinct purple-black margins. Heavy vein discoloration (dark brown) within lesions distinguishes from Cercospora. **Pod/Stem Phase:** Pods develop circular, sunken black lesions often covered with salmon-pink spore masses during humid mornings. Stems show elongated, reddish-brown cankers that may girdle plants. **Seedling Infections:** Cotyledons display dark, sunken spots leading to damping-off. Key differentiation: Anthracnose lesions are more angular and darker than Brown Spot's rounded, lighter lesions.",
            "risk_impact": "**Yield Reduction:** Moderate foliar infection = 10-25% loss. Severe defoliation + pod infection = 50-80% loss. Plants may die prematurely if stem girdling occurs. **Seed Quality Degradation:** Infected seeds show 40-60% germination (vs 90% healthy), reduced vigor, and 15-20% lower oil content. Shriveled seeds fail USDA Grade 1 standards. **Economic Impact:** Combined yield + quality losses = $80-150/acre in severe outbreaks. Seed lots with >5% infected seeds rejected by processors. Replanting costs $40-60/acre if early-season stand loss exceeds 20%.",
            "control_strategy": "**Fungicide Strategy:** Deploy preventive program: (1) R3 stage—apply strobilurin fungicide (azoxystrobin 6.2 fl oz/acre), (2) R5 stage—apply DMI + strobilurin mix (prothioconazole + trifloxystrobin) if wet weather continues. Add spreader-sticker to improve leaf coverage. **Varietal Resistance:** Select MG III-IV varieties with Rcs3 or Rpp? resistance genes (consult seed company ratings). **Field Sanitation:** Immediately after harvest, shred stalks and disk residue to accelerate decomposition. Avoid minimum-till in fields with >20% infection history.",
            "monitoring_prevention": "**Critical Monitoring Period:** Days 3-10 post-detection are critical. Inspect 50 plants per field section (corner, center, edge) daily. Count lesions per plant—threshold for treatment: ≥5 lesions on upper canopy or any pod lesions. **Weather-Based Action:** Install weather sensor or use NearCast app to track leaf wetness duration. Trigger spray when forecast shows 12+ hours leaf wetness in next 72 hours. **Preventive Measures:** Next season: (1) use certified Anthracnose-free seed, (2) rotate to non-host crops for 24+ months, (3) plant early to escape late-season wet periods, (4) maintain pH 6.0-6.8 to optimize plant defense responses."
        },
        {
            "variant_focus": "Field-Wide vs Plant-Level Management",
            "insight": "*Colletotrichum truncatum* spreads via splash-dispersed conidia (short-range, 3-10 feet) and wind-borne ascospores (long-range, miles). This dual-dispersal biology requires both field-scale and individual plant management. Overwintered inoculum in crop debris initiates primary infections during spring rains. Secondary spread accelerates when canopy closure traps humidity (typically R3-R5). High-density plantings (>180,000 plants/acre) increase within-field spread rates by 40-60%. Adjacent fields with infected residue serve as continuous spore sources throughout the season.",
            "visual_identification": "**Field-Level Patterns:** Infection initially appears as scattered foci (circular patches 5-15 feet diameter) in low-lying areas with poor drainage. As season progresses, patches coalesce from field edges inward. Use aerial scouting (drone or high clearance vehicle) to map distribution. **Plant-Level Symptoms:** Individual plants show progressive lower-to-upper canopy infection. Lesions first appear on lowest trifoliate leaves (3-7 days post-infection), then move upward 1-2 leaf tiers per week. Mature lesions (7-14 days old) display characteristic angular shape with dark margins and tan centers. **Critical Differentiator:** Anthracnose's stem cankers (elongated, sunken, dark brown) distinguish it from leaf-only pathogens like Cercospora.",
            "risk_impact": "**Field-Scale Impact:** Uniform infections across entire fields indicate high inoculum pressure—expect 30-50% yield loss without intervention. Patchy infections (affecting <40% of field area) result in 10-25% overall loss. **Plant-Level Consequences:** Individual heavily infected plants (>50% defoliation) rarely produce marketable seed. Neighboring plants within 5-foot radius face 3-5x higher infection probability due to microclimate effects (humidity, spore concentration). **Long-Term Field Health:** Fields with severe Anthracnose history develop 'disease suppressive' status over 3-4 years if continuously rotated, reducing future outbreak severity by 60-75%.",
            "control_strategy": "**Field-Wide Strategy:** For uniform high-pressure infections, apply fungicide via ground rig or aerial application to entire field. Use high-volume spray (15-20 GPA) for thorough canopy penetration. Prioritize fields in reproductive stages (R3+) over vegetative fields. **Targeted Plant Management:** In patchy infections, treat only affected zones plus 30-foot buffer. Flag individual highly diseased plants for removal if economically feasible (e.g., seed production fields). **Integrated Approach:** Combine cultural practices (widen rows to 30 inches, avoid late planting dates) with 2-3 fungicide applications timed to growth stages and weather forecasts.",
            "monitoring_prevention": "**Field-Level Surveillance:** Scout entire field perimeter weekly—disease often enters from edges near infected residue or adjacent fields. Use GPS to mark initial infection foci; monitor expansion rate (measure patch diameter weekly). If expansion >5 feet/week, implement immediate chemical control. **Plant-Level Monitoring:** Within infection zones, assess 10 plants per focal point—calculate disease severity index (% defoliation × incidence). Treat if index >15. **Prevention for Next Season:** (1) Conduct post-harvest soil assays to quantify *Colletotrichum* populations. (2) High-risk fields (>1000 CFU/g soil) require 3-year soybean exclusion. (3) Plant windbreak varieties (taller, early-maturing MG II-III) on field edges to intercept airborne spores before reaching commercial sections."
        },
        {
            "variant_focus": "Integrated Disease Suppression",
            "insight": "Anthracnose management requires integrating host resistance, environmental modification, and targeted chemical intervention. *Colletotrichum truncatum* overwinters as microsclerotia in soil and infected residue, surviving 2-3 years in temperate climates. Germination occurs when spring temperatures stabilize at 70-80°F with rainfall events >0.5 inches. The pathogen exhibits race-specific virulence—certain races overcome host resistance genes rapidly (within 2-3 growing seasons). Successful suppression combines partial resistance (Rcs3 gene provides 40-60% protection), microclimate management (reducing canopy humidity), and protectant fungicides during critical infection windows (R3, R5 growth stages).",
            "visual_identification": "**Progression Timeline:** Day 1-3 post-infection: Tiny water-soaked spots (1-2mm), easily missed without magnification. Day 4-7: Lesions expand to 5-8mm with defined borders and center necrosis. Day 8-14: Mature lesions reach 10-20mm, often angular, following leaf veins. Dark brown to black margins with tan-gray centers. **Advanced Symptoms:** By Day 15+, multiple lesions coalesce, causing entire leaflets to necrose and drop. Petioles show dark streaking. Pods develop characteristic black, sunken lesions with acervuli (small black spore-producing structures) visible with 10x lens. **Diagnostic Features:** Unlike Frogeye Leaf Spot (concentric rings), Anthracnose lesions are irregular. Unlike Target Spot (bulls-eye pattern), they lack concentric zonation. Stem cankers clinch diagnosis.",
            "risk_impact": "**Tiered Risk Assessment:** Low severity (5-15% leaf area affected) = 5-10% yield loss, minimal quality impact. Moderate severity (15-40% leaf area + some pod infection) = 20-35% yield loss, 10-20% seed quality degradation. High severity (>40% defoliation, extensive pod infection, stem girdling) = 50-80% yield loss, 40-60% seed rejection, potential stand collapse. **Market Implications:** Grain elevators dock 2-5 cents/bushel for each 1% increase in damage beyond 3%. Seed producers face contract cancellations if infection exceeds 5% incidence. **Epidemiological Impact:** Heavily infected fields become regional inoculum sources, increasing neighboring farms' spray costs by $30-50/acre over subsequent 2 seasons.",
            "control_strategy": "**Pre-Season Foundation:** Select varieties with highest Anthracnose tolerance ratings (≥7/10 on seed company scales). Test seed lots for latent infection before planting—reject lots with >0.5% infected seeds. Treat all seeds with fungicide + insecticide combination (metalaxyl + thiamethoxam). **In-Season Layered Protection:** (1) V6-R1: Monitor weather—if 3+ days of leaf wetness forecasted, apply preventive azoxystrobin (4-6 oz/acre). (2) R3: Apply protectant DMI fungicide (tebuconazole 4 oz/acre). (3) R5: If infection detected or humid weather persists, apply strobilurin + DMI tank mix. **Resistance Management:** Never exceed 2 FRAC 11 (strobilurin) applications per season. Rotate with FRAC 3 (DMI) and FRAC 7 (SDHI) chemistries.",
            "monitoring_prevention": "**Intensive Monitoring Protocol:** Days 1-7: Inspect lower canopy (most humid zone) daily for first lesions. Use 10x hand lens—early detection enables preventive action vs reactive spraying. Establish 5 permanent monitoring stations per 40 acres. **Week 2-4:** Sample 20 leaflets per station weekly. Calculate percent leaf area diseased using smartphone apps (Assess or Leaf Doctor). Threshold for treatment: 5% diseased leaf area or any pod lesions. Monitor weather: <85°F and <12 hours leaf wetness = low risk; >85°F and >16 hours wetness = extreme risk. **Long-Term Prevention:** Post-harvest: Sample soil from 5 field locations (0-6 inch depth), send to diagnostic lab for *Colletotrichum* quantification. Fields with >500 propagules/g require 2-year rotation. Implement cover crops (cereal rye, wheat) to competitively suppress fungal survival. Avoid volunteer soybeans—they perpetuate inoculum despite crop rotation."
        }
    ],
    
    "Healthy": [
        {
            "variant_focus": "Optimal Health Maintenance",
            "insight": "Your soybean plant exhibits no detectable disease symptoms, indicating successful integrated crop management. Healthy plants maintain dark green foliage with uniform leaf coverage, intact cuticles, and proper nodulation (10-20 nitrogen-fixing nodules per root system). This status results from effective combination of disease-resistant variety selection, balanced soil nutrition (optimal pH 6.0-6.8, adequate Ca/Mg/K ratios), and favorable environmental conditions. Continued health requires proactive monitoring as disease pressure fluctuates seasonally with temperature, humidity, and spore load from surrounding fields.",
            "visual_identification": "**Confirmation Indicators:** Leaves show uniform dark green coloration without spots, lesions, or discoloration. Leaf margins are smooth without necrosis or curling. Trifoliate leaves fully expanded with no stunting. Stems are firm, green (or normal brown for mature plants), without cankers, galls, or unusual swelling. Petioles intact without lesions or dark streaking. **Growth Stage Assessment:** Vegetative plants (V4-V6) should have 10-14 fully developed trifoliates. Reproductive plants (R3-R5) should have abundant flower/pod set with no premature drop. **Diagnostic Absence:** No symptoms of common diseases—no leaf spots (Anthracnose, Cercospora, Brown Spot), no powdery mildew coating, no mosaic patterns, no root rot indicators (wilting, yellowing). Plant vigor aligns with variety's expected growth pattern.",
            "risk_impact": "**Current Status:** Zero disease-related yield loss. Plant is achieving genetic yield potential (40-60 bushels/acre for standard varieties, 60-80+ for high-yield genetics). Photosynthetic capacity at maximum (~6-8 µmol CO₂/m²/s at peak midday). Seed quality projected to meet Grade 1 standards (>85% germination, <2% damage, <13% moisture). **Maintenance Value:** Preserving health avoids $30-80/acre in treatment costs and $50-200/acre in yield losses typical of diseased fields. **Future Risk:** Healthy status is not permanent—disease pressure increases during R3-R6 if environmental conditions favor pathogens (extended humidity >85%, rainfall >0.5 inch/event, temperatures 75-85°F).",
            "control_strategy": "**Preventive Protection:** Continue current management practices—maintain whatever disease prevention measures brought plant to this status (resistant varieties, proper rotation, balanced fertility). Monitor weather forecasts: if extended humid periods (>3 days with >90% RH overnight) are predicted during reproductive stages, consider prophylactic fungicide (strobilurin class, 4-6 oz/acre) as insurance. **Nutritional Support:** Conduct mid-season tissue testing to ensure nutrient levels remain optimal: N 4.5-5.5%, P 0.3-0.5%, K 1.7-2.5%, S 0.25-0.4%. Apply foliar micronutrients (Mn, Fe, Zn) if deficiencies detected. **Biological Enhancement:** Maintain soil microbial health through avoiding excessive tillage and considering cover crops next season. Healthy soil biology suppresses pathogen populations naturally.",
            "monitoring_prevention": "**Ongoing Surveillance:** Healthy plants still require weekly scouting—diseases can emerge rapidly under favorable conditions. Inspect 20-30 plants per field section (corners, edges, center) focusing on lower canopy where humidity is highest and disease initiates. Use 10x hand lens to detect early lesions (2-3mm) before they become visible to naked eye. **Proactive Thresholds:** If scouting detects ≥1 lesion per 10 lower leaves, consider preventive fungicide even if severity is low—early intervention prevents epidemic development. **Seasonal Vigilance:** Critical monitoring periods: (1) R3 (beginning pod) when canopy closes and humidity increases, (2) R5 (seed fill) when plants are most susceptible due to resource allocation to seeds. **Next Season Preparation:** Document current practices—variety, planting date, fertility program, pest management—to replicate success. Save seed lot information for future variety selection. Consider participating in yield contests or farmer networks to benchmark your success."
        },
        {
            "variant_focus": "Proactive Risk Management",
            "insight": "Current absence of disease does not guarantee season-long health. Soybean diseases are often latent—pathogens may be present in field (overwintered spores, infected residue) without causing visible symptoms until environmental triggers occur. Healthy status indicates successful early-season management: appropriate variety selection (resistance genes to prevalent local pathogens), optimal planting date (avoiding early-season wet periods), proper seed treatment (fungicide + insecticide protection), and favorable weather (dry conditions during establishment). Maintaining health requires understanding dynamic disease risk: spore loads peak during R3-R5 when neighboring fields shed inoculum, and plant defenses decline during pod fill when resources divert to seed production.",
            "visual_identification": "**Baseline Health Markers:** Leaves exhibit proper turgor (firm, not wilted), indicating adequate soil moisture and functional root systems. Chlorophyll content high (SPAD meter readings 38-42 for mid-season plants). Internode length consistent with variety characteristics (2-3 inches for standard types). **Negative Diagnostics:** No signs of: (1) Fungal pathogens—no leaf spots, stem cankers, or root discoloration; (2) Bacterial infections—no water-soaked lesions or systemic wilt; (3) Viral diseases—no mosaic patterns, leaf distortion, or stunting; (4) Nutrient deficiencies—no chlorosis, necrosis, or abnormal coloration; (5) Insect damage—no feeding scars, stippling, or defoliation. **Early Detection Readiness:** Establish this healthy appearance as your baseline reference for detecting subtle changes during subsequent scouting trips.",
            "risk_impact": "**Value Preservation:** Healthy plants represent $400-800/acre investment (land rent + seed + inputs) with projected return of $800-1200/acre at 50-60 bu/acre yields. Disease outbreaks can erase 25-75% of this return ($200-600/acre loss) depending on pathogen, timing, and severity. **Regional Context:** Even healthy fields face external risk—spores travel miles on wind, and one infected field can inoculate entire counties during wet weather. Your field's disease-free status provides community benefit by not contributing to regional spore loads. **Insurance Opportunity:** Healthy mid-season status creates option for modest fungicide investment ($15-25/acre product + application) to protect against late-season disease surges. ROI typically 2-5:1 if infection prevented during critical R5-R6 pod fill stages.",
            "control_strategy": "**Strategic Protection:** For currently healthy fields entering reproductive stages, evaluate preventive fungicide economics: In high-pressure years (wet, humid seasons with disease reports in surrounding counties), prophylactic application at R3 provides 4-8 bushel yield protection (10-20% ROI). Use USDA's ipmPIPE or state extension disease forecasting tools to assess regional risk. **Variety Validation:** Document your variety's performance—if it remains healthy while neighbors' fields develop disease, prioritize this variety in future plantings. Contact seed company to understand its resistance profile. **Integrated Management:** Healthy status enables flexibility—can delay fungicide decision until R3-R4 based on weather forecasts rather than reactive spraying. This precision timing optimizes efficacy and ROI.",
            "monitoring_prevention": "**Risk-Based Scouting:** Healthy fields require less frequent intensive scouting but shouldn't be ignored. Scout every 7-10 days (vs every 3-5 days for at-risk fields). Focus on: (1) Field edges adjacent to previous soybean fields (high inoculum reservoirs), (2) Low-lying areas with extended leaf wetness, (3) Dense canopy sections with reduced air circulation. **Weather Monitoring:** Use NearCast, Greencast, or similar apps to track leaf wetness duration and infection risk ratings. High-risk periods (Favorable weather for 3+ consecutive days) warrant extra scouting trip. **Preventive Timing:** If maintaining health is goal, consider these fungicide timing options based on disease pressure: (a) Low pressure—no spray, monitor only; (b) Moderate pressure—single R3 application; (c) High pressure—R3 + R5 applications. **Next Year Insurance:** Even if no disease occurred this year, rotate away from soybeans for 1-2 years to prevent inoculum buildup. Plant corn, wheat, or cover crops to break disease cycles. Select different MG or variety next soybean cycle to avoid variety-specific pathogens."
        }
    ]
    # Additional diseases would follow the same pattern with 5 variants each
}

def get_random_knowledge_variant(disease_name: str) -> Optional[Dict[str, str]]:
    """
    Get a random knowledge variant for the specified disease.
    Returns None if disease has no variants defined.
    """
    variants = DISEASE_KNOWLEDGE_VARIANTS.get(disease_name, [])
    if not variants:
        logger.warning(f"No knowledge variants found for disease: {disease_name}")
        return None
    
    selected = random.choice(variants)
    logger.info(f"Selected variant '{selected['variant_focus']}' for {disease_name}")
    return selected

class DynamicLLMReasoningLayer:
    """
    Generates intelligent, context-aware agricultural advice using disease predictions.
    Runs AFTER CNN classification to enhance explanations without changing predictions.
    """
    
    def __init__(self, llm_backend='local', model_name='template'):
        """
        Initialize LLM reasoning layer.
        
        Args:
            llm_backend: 'local' (LLaMA/Mistral), 'api' (OpenAI/Anthropic), or 'template'
            model_name: Specific model identifier
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        
        # Start with template-based reasoning (can upgrade to real LLM later)
        if llm_backend == 'template':
            logger.info("Using template-based reasoning (upgradeable to LLaMA/Mistral)")
        else:
            logger.info(f"LLM backend: {llm_backend}, Model: {model_name}")
    
    def generate_dynamic_advice(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method: Generate context-aware advice based on CNN prediction.
        
        Args:
            prediction_data: {
                'crop': str,
                'disease': str,
                'confidence': float (0-1),
                'severity': str,
                'static_knowledge': {
                    'symptoms': str,
                    'precautions': str,
                    'treatment': str,
                    'fertilizers': str,
                    ...
                }
            }
        
        Returns:
            Enhanced advice with dynamic explanations
        """
        try:
            # Validate input
            if not self._validate_input(prediction_data):
                logger.warning("Invalid prediction data, falling back to static")
                return self._fallback_static(prediction_data)
            
            # Build LLM prompt
            prompt = self._build_prompt(prediction_data)
            
            # Generate reasoning
            if self.llm_backend == 'template':
                reasoning = self._template_based_reasoning(prediction_data)
            elif self.llm_backend == 'local':
                reasoning = self._local_llm_reasoning(prompt, prediction_data)
            elif self.llm_backend == 'api':
                reasoning = self._api_llm_reasoning(prompt, prediction_data)
            else:
                reasoning = self._template_based_reasoning(prediction_data)
            
            # Safety check: ensure no prediction override
            reasoning['disease'] = prediction_data['disease']
            reasoning['confidence'] = prediction_data['confidence']
            
            logger.info(f"Generated dynamic advice for {prediction_data['disease']}")
            return reasoning
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}. Using fallback.")
            return self._fallback_static(prediction_data)
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate prediction data structure"""
        required = ['disease', 'confidence']
        return all(key in data for key in required)
    
    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """
        Build structured prompt for LLM.
        
        Template format ensures consistent, safe outputs.
        """
        crop = data.get('crop', 'Soybean')
        disease = data['disease']
        confidence = data['confidence']
        severity = data.get('severity', 'Unknown')
        static = data.get('static_knowledge', {})
        
        prompt = f"""You are an expert agricultural advisor analyzing {crop} disease detection results.

DETECTION RESULTS:
- Crop: {crop}
- Disease Detected: {disease}
- Confidence: {confidence:.1%}
- Severity Level: {severity}

EXISTING KNOWLEDGE BASE:
{json.dumps(static, indent=2)}

YOUR TASKS:
1. Provide a clear, farmer-friendly explanation of {disease}
2. Adjust advice tone based on confidence ({confidence:.1%}) and severity ({severity})
3. Give immediate actionable steps
4. Contextualize fertilizer recommendations based on severity
5. Suggest monitoring activities for next few days

CONSTRAINTS:
- Do NOT change the disease name
- Do NOT contradict knowledge base facts
- Do NOT recommend unsafe chemical dosages
- Use simple language suitable for farmers
- Be concise but thorough

Generate a structured response."""

        return prompt
    
    def _template_based_reasoning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template-based intelligent reasoning with variant knowledge system.
        Randomly selects from 5+ expert-curated knowledge variants per disease.
        """
        disease = data['disease']
        confidence = data['confidence']
        severity = data.get('severity', 'Unknown')
        static = data.get('static_knowledge', {})
        
        # Try to get variant knowledge for this disease
        variant = get_random_knowledge_variant(disease)
        
        # If variant exists, use its structured knowledge
        if variant:
            logger.info(f"Using variant knowledge system for {disease} (focus: {variant['variant_focus']})")
            return self._build_variant_response(disease, confidence, severity, variant)
        
        # Fallback to original template-based system if no variants defined
        logger.info(f"No variants for {disease}, using standard template system")
        return self._build_standard_template_response(disease, confidence, severity, static)
    
    def _build_variant_response(self, disease: str, confidence: float, severity: str, variant: Dict[str, str]) -> Dict[str, Any]:
        """Build response using variant knowledge structure"""
        
        # Confidence-based tone
        if confidence >= 0.90:
            confidence_phrase = "with high certainty"
        elif confidence >= 0.70:
            confidence_phrase = "with good confidence"
        else:
            confidence_phrase = "with moderate confidence"
        
        # Build dynamic explanation incorporating variant insight
        if disease.lower() == 'healthy':
            urgency = "STATUS"
            explanation_header = f"**PLANT STATUS - Healthy** (Detection Accuracy: {confidence:.1%})"
        else:
            urgency = "ALERT"
            explanation_header = f"**{urgency} - {disease} Detected** {confidence_phrase} (Detection Accuracy: {confidence:.1%})"
        
        dynamic_explanation = f"""{explanation_header}

**Disease Insight:**
{variant['insight']}

**Current Severity:** {severity}
"""
        
        # Assemble full response from variant sections
        return {
            'disease': disease,
            'confidence': confidence,
            'severity': severity,
            
            # Variant-based dynamic content
            'dynamic_explanation': dynamic_explanation,
            'symptoms': variant['visual_identification'],
            'impact_analysis': variant['risk_impact'],
            'treatment': variant['control_strategy'],
            'monitoring_plan': variant['monitoring_prevention'],
            
            # Metadata
            'reasoning_method': 'variant_knowledge',
            'variant_focus': variant['variant_focus'],
            'confidence_level': 'high' if confidence >= 0.90 else 'moderate' if confidence >= 0.70 else 'low'
        }
    
    def _build_standard_template_response(self, disease: str, confidence: float, severity: str, static: Dict) -> Dict[str, Any]:
        """Original template-based system (fallback for diseases without variants)"""
        
        # Confidence-based tone adjustment
        if confidence >= 0.90:
            confidence_phrase = "with high certainty"
            action_urgency = "immediate action is recommended"
        elif confidence >= 0.70:
            confidence_phrase = "with good confidence"
            action_urgency = "prompt attention is advised"
        else:
            confidence_phrase = "with moderate confidence"
            action_urgency = "further monitoring and verification recommended"
        
        # Severity-based advice adjustment
        severity_lower = severity.lower() if severity else 'unknown'
        
        if disease.lower() == 'healthy':
            urgency = "STATUS"
            treatment_tone = "Your plant shows no signs of disease. Keep up the good work!"
            fertilizer_note = "Continue with your regular fertilization schedule."
            dynamic_explanation = f"""**PLANT STATUS - Healthy**

The AI system has verified that your plant is **Healthy** {confidence_phrase} (Detection Accuracy: {confidence:.1%}).

**What This Means:**
Your soybean plant is in good health and shows no significant signs of the diseases currently monitored by the system.

**Next Steps:**
Continue with standard care and regular monitoring to maintain this health level."""
        elif 'high' in severity_lower or 'severe' in severity_lower:
            urgency = "URGENT"
            treatment_tone = "Immediate treatment is critical to prevent significant crop loss."
            fertilizer_note = "Apply recommended fertilizers at standard rates to support plant recovery."
        elif 'moderate' in severity_lower or 'medium' in severity_lower:
            urgency = "IMPORTANT"
            treatment_tone = "Timely intervention will help control disease spread and minimize damage."
            fertilizer_note = "Standard fertilizer application will support plant health during recovery."
        else:
            urgency = "ROUTINE"
            treatment_tone = "Early-stage detection allows for preventive measures."
            fertilizer_note = "Maintain balanced fertilization to strengthen plant immunity."
        
        # Generate dynamic explanation
        dynamic_explanation = f"""**{urgency} ALERT - {disease} Detected**

The AI system has identified **{disease}** {confidence_phrase} (Detection Accuracy: {confidence:.1%}).

**What This Means:**
{static.get('meaning', disease + ' is a disease affecting soybean plants.')}

**Current Severity:** {severity}
{treatment_tone}

**Why You Should Act Now:**
{action_urgency.capitalize()}. Early intervention significantly improves treatment success rates."""
        
        # Dynamic immediate actions
        immediate_steps = self._generate_immediate_actions(disease, severity, static)
        
        # Contextual monitoring plan
        monitoring_plan = self._generate_monitoring_plan(disease, severity, confidence)
        
        # Severity-adjusted fertilizer advice
        fertilizer_advice = self._adjust_fertilizer_advice(static.get('fertilizers', ''), 
                                                           severity, 
                                                           fertilizer_note)
        
        # Assemble enhanced response
        enhanced_response = {
            'disease': disease,  # NEVER CHANGED
            'confidence': confidence,  # NEVER CHANGED
            'severity': severity,
            
            # Dynamic sections
            'dynamic_explanation': dynamic_explanation,
            'immediate_actions': immediate_steps,
            'monitoring_plan': monitoring_plan,
            'fertilizer_guidance': fertilizer_advice,
            
            # Original static data (preserved)
            'symptoms': static.get('symptoms', 'No symptom data'),
            'precautions': static.get('precautions', 'No precaution data'),
            'treatment': static.get('treatment', 'No treatment data'),
            'fertilizers': static.get('fertilizers', 'No fertilizer data'),
            
            # Metadata
            'reasoning_method': 'template_based',
            'confidence_level': 'high' if confidence >= 0.90 else 'moderate' if confidence >= 0.70 else 'low'
        }
        
        return enhanced_response
    
    def _generate_immediate_actions(self, disease: str, severity: str, static: Dict) -> str:
        """Generate actionable immediate steps based on context"""
        base_treatment = static.get('treatment', '')
        
        # Extract key actions from treatment text
        actions = []
        
        if severity and ('high' in severity.lower() or 'severe' in severity.lower()):
            actions.append("1. **Isolate affected plants** immediately to prevent disease spread")
            actions.append("2. **Remove and destroy** severely infected leaves following biosafety protocols")
            actions.append("3. **Apply recommended fungicide/bactericide** as per treatment guidelines")
            actions.append("4. **Inspect neighboring plants** within 5-meter radius daily")
        else:
            actions.append("1. **Mark affected plants** for monitoring")
            actions.append("2. **Apply preventive treatment** as recommended")
            actions.append("3. **Improve air circulation** around plants if possible")
            actions.append("4. **Monitor daily** for symptom progression")
        
        actions.append(f"5. **Document observations** including photo records for tracking")
        
        return "\n".join(actions)
    
    def _generate_monitoring_plan(self, disease: str, severity: str, confidence: float) -> str:
        """Generate contextual monitoring recommendations"""
        
        if confidence < 0.70:
            primary = "**Verify detection results:** Capture additional clear images of affected leaves from multiple angles for re-analysis."
        else:
            primary = "**Track symptom progression:** Document daily changes in leaf appearance and disease spread patterns."
        
        plan = f"""{primary}

**Next 3 Days:**
- Inspect plants every morning and evening
- Record number of newly affected leaves
- Check for symptom intensification or improvement
- Monitor weather conditions (humidity, rainfall)

**Next 1-2 Weeks:**
- Evaluate treatment effectiveness
- Adjust intervention strategy if needed
- Consider crop health improvement or deterioration trends
- Prepare for potential follow-up treatments

**Seek Expert Help If:**
- Disease spreads rapidly despite treatment
- New unusual symptoms appear
- Treatment shows no improvement after 1 week
- Confidence in detection was low (<70%) and symptoms worsen"""
        
        return plan
    
    def _adjust_fertilizer_advice(self, base_fertilizers: str, severity: str, context_note: str) -> str:
        """Contextualize fertilizer recommendations based on severity"""
        
        if not base_fertilizers or base_fertilizers == 'No fertilizer data':
            return "Consult local agricultural extension for fertilizer recommendations specific to your soil conditions."
        
        advice = f"""**Contextual Fertilizer Guidance:**

{context_note}

**Recommended Nutrients (from knowledge base):**
{base_fertilizers}

**Application Guidelines:**
"""
        
        severity_lower = severity.lower() if severity else ''
        
        if 'high' in severity_lower or 'severe' in severity_lower:
            advice += """- Apply at **standard recommended rates** - do not over-apply
- Focus on **balanced NPK** to support recovery without stressing plants
- Consider **foliar micronutrients** for faster absorption
- **Split applications** if total dose is high (reduce stress)"""
        else:
            advice += """- Follow **standard application rates** for your crop stage
- Maintain **regular fertilization schedule**
- Emphasize **potassium and micronutrients** for disease resistance
- Avoid **nitrogen excess** which can increase disease susceptibility"""
        
        advice += "\n\n⚠️ **Warning:** Always perform soil tests before major fertilizer adjustments."
        
        return advice
    
    def _local_llm_reasoning(self, prompt: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for local LLM integration (LLaMA/Mistral).
        Can be implemented when local LLM is available.
        """
        logger.info("Local LLM not yet configured, using template fallback")
        return self._template_based_reasoning(data)
    
    def _api_llm_reasoning(self, prompt: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for API-based LLM integration (OpenAI/Anthropic).
        Can be implemented when API access is configured.
        """
        logger.info("API LLM not yet configured, using template fallback")
        return self._template_based_reasoning(data)
    
    def _fallback_static(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safety fallback: return static data if LLM reasoning fails.
        Ensures system always provides output.
        """
        static = data.get('static_knowledge', {})
        
        return {
            'disease': data.get('disease', 'Unknown'),
            'confidence': data.get('confidence', 0.0),
            'severity': data.get('severity', 'Unknown'),
            'dynamic_explanation': f"Disease detected: {data.get('disease', 'Unknown')}",
            'symptoms': static.get('symptoms', 'No data'),
            'precautions': static.get('precautions', 'No data'),
            'treatment': static.get('treatment', 'No data'),
            'fertilizers': static.get('fertilizers', 'No data'),
            'immediate_actions': 'Consult static treatment recommendations',
            'monitoring_plan': 'Monitor plant health regularly',
            'fertilizer_guidance': static.get('fertilizers', 'No data'),
            'reasoning_method': 'static_fallback'
        }


# Singleton instance
_llm_reasoning_instance = None

def get_llm_reasoning_layer(llm_backend='template', model_name='template'):
    """
    Get singleton instance of LLM reasoning layer.
    
    Args:
        llm_backend: 'template', 'local', or 'api'
        model_name: Model identifier
    
    Returns:
        DynamicLLMReasoningLayer instance
    """
    global _llm_reasoning_instance
    
    if _llm_reasoning_instance is None:
        _llm_reasoning_instance = DynamicLLMReasoningLayer(
            llm_backend=llm_backend,
            model_name=model_name
        )
    
    return _llm_reasoning_instance


def enhance_with_llm_reasoning(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to enhance predictions with LLM reasoning.
    
    Args:
        prediction_data: CNN prediction + static knowledge
    
    Returns:
        Enhanced advice with dynamic explanations
    """
    llm_layer = get_llm_reasoning_layer()
    return llm_layer.generate_dynamic_advice(prediction_data)
