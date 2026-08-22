"""Enhanced Soybean Disease Knowledge Dataset (LLM-Ready)
Comprehensive agricultural information for AI-powered disease detection system
"""

DISEASE_KNOWLEDGE = {
    'Bacterial Blight': {
        'disease_type': {
            'classification': 'Bacterial Disease',
            'causal_organism': 'Pseudomonas syringae pv. glycinea',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Bacterial blight presents as distinctive water-soaked lesions that rapidly develop into angular brown spots with prominent yellow halos.',
            'visual_indicators': [
                'Small water-soaked spots on leaves (initial stage)',
                'Angular brown lesions with yellow halos (advanced)',
                'Leaf tissue tearing and premature leaf drop',
                'Stunted plant growth and reduced vigor',
                'Dark brown stem lesions in severe cases'
            ],
            'affected_parts': ['Leaves (primary)', 'Stems', 'Petioles'],
            'progression': 'Appears first on lower leaves, spreads upward with rain splash and wind'
        },
        'precautions': {
            'seed_management': 'Use certified disease-free seeds from reputable sources',
            'cultural_practices': [
                'Implement 2-3 year crop rotation with non-legume crops',
                'Avoid overhead irrigation; use drip or furrow irrigation',
                'Maintain field sanitation by removing infected plant debris',
                'Ensure adequate plant spacing (15-20 inches) for air circulation',
                'Avoid working in fields when plants are wet'
            ],
            'resistant_varieties': 'Select varieties with resistance genes like Rpg1-b, Rpg2, Rpg3',
            'monitoring': 'Scout fields weekly during vegetative stages, especially after rainfall'
        },
        'treatment': {
            'chemical_control': {
                'products': ['Copper hydroxide', 'Copper sulfate', 'Copper oxychloride'],
                'application_timing': 'Apply at first sign of symptoms; repeat every 7-10 days',
                'efficacy_note': 'Bactericides reduce spread but do not cure infected tissue'
            },
            'cultural_control': [
                'Remove and destroy severely infected plants immediately',
                'Burn or bury infected plant material away from field',
                'Disinfect farm equipment between fields'
            ],
            'biological_control': 'Bacillus subtilis-based products can suppress bacterial populations',
            'prognosis': 'No complete cure once established; management focuses on limiting spread'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced formula 15-15-15 or as per soil test',
                'nitrogen': 'Moderate N (40-60 kg/ha); excess promotes disease',
                'phosphorus': '60-80 kg P₂O₅/ha for root strength',
                'potassium': '40-60 kg K₂O/ha for disease resistance'
            },
            'micronutrients': {
                'zinc': '5-10 kg ZnSO₄/ha to strengthen leaf tissue',
                'boron': '0.5-1.0 kg/ha for cell wall integrity'
            },
            'organic_amendments': [
                'Well-decomposed compost (5-10 tons/ha)',
                'Vermicompost (2-3 tons/ha)',
                'Neem cake (200-300 kg/ha) for disease suppression'
            ],
            'soil_management': 'Maintain soil pH 6.0-6.8 for optimal nutrient availability'
        },
        'management_strategies': {
            'preventive_measures': [
                'Plant resistant varieties with Rpg genes',
                'Use certified disease-free seeds',
                'Implement 2-3 year crop rotation',
                'Avoid overhead irrigation',
                'Ensure proper field drainage'
            ],
            'early_detection': [
                'Monitor fields during warm, humid conditions',
                'Look for water-soaked lesions on lower leaves',
                'Check for angular brown spots with yellow halos',
                'Watch for premature defoliation'
            ],
            'integrated_approach': [
                'Combine cultural, biological, and chemical controls',
                'Focus on reducing inoculum sources',
                'Time applications appropriately',
                'Use resistant varieties as primary defense'
            ]
        },
        'economic_impact': 'Can cause 10-40% yield loss in susceptible varieties under favorable conditions',
        'environmental_conditions': {
            'favorable': 'Temperature 24-28°C, high humidity (>80%), frequent rainfall',
            'unfavorable': 'Hot dry conditions slow disease development'
        },
        'research_advances': [
            'New resistant varieties with stacked Rpg genes',
            'Improved copper formulations with better efficacy',
            'Biological control agents with enhanced effectiveness',
            'Precision agriculture for targeted application'
        ]
    },

    'Anthracnose': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Colletotrichum truncatum',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Anthracnose typically appears during the reproductive stages, causing irregular brown necrotic lesions on stems, pods, and leaves.',
            'visual_indicators': [
                'Irregularly shaped brown areas on stems and pods',
                'Premature defoliation',
                'Small black fruiting bodies (acervuli) on infected areas',
                'Reduced seed quality and quantity'
            ],
            'affected_parts': ['Stems', 'Pods', 'Leaves'],
            'progression': 'Spreads rapidly in warm, humid weather, especially during pod-filling stages.'
        },
        'precautions': {
            'seed_management': 'Use certified disease-free seeds and fungicide seed treatments.',
            'cultural_practices': [
                'Crop rotation with non-host crops like corn',
                'Ensure proper drainage',
                'Remove or bury crop residue after harvest'
            ],
            'resistant_varieties': 'Select varieties with known tolerance to Anthracnose.',
            'monitoring': 'Monitor fields closely during R1 to R5 stages.'
        },
        'treatment': {
            'chemical_control': {
                'products': ['Strobilurins', 'Triazoles', 'Benzimidazoles'],
                'application_timing': 'Apply at R1-R3 stages if symptoms appear.',
                'efficacy_note': 'Fungicides are most effective when applied preventively.'
            },
            'cultural_control': [
                'Destroy infected crop residue',
                'Deep plowing to bury infected material'
            ],
            'biological_control': 'Some Bacillus-based products show suppression.',
            'prognosis': 'Can significantly reduce yield if left unmanaged during reproductive stages.'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'potassium': 'High K levels improve stem strength and disease resistance.'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Use anthracnose-resistant varieties',
                'Apply fungicide seed treatments',
                'Implement 2-3 year crop rotation',
                'Monitor fields during reproductive stages',
                'Remove infected crop debris'
            ],
            'early_detection': [
                'Inspect stems and pods for brown lesions',
                'Look for small black fruiting bodies',
                'Monitor during warm, humid weather',
                'Check for premature defoliation'
            ],
            'integrated_approach': [
                'Combine resistant varieties with fungicide applications',
                'Time fungicide applications at R1-R3 stages',
                'Maintain field sanitation',
                'Use cultural practices to reduce inoculum'
            ]
        },
        'economic_impact': 'Yield losses of 10-30% in susceptible varieties during favorable conditions',
        'environmental_conditions': {
            'favorable': 'Warm temperatures (24-30°C), high humidity, prolonged leaf wetness',
            'unfavorable': 'Dry conditions, temperatures above 32°C'
        },
        'research_advances': [
            'New resistant varieties with multiple resistance genes',
            'Advanced fungicide formulations with extended residual',
            'Precision application technologies',
            'Molecular markers for early detection'
        ]
    },

    'Bacterial Pustule': {
        'disease_type': {
            'classification': 'Bacterial Disease',
            'causal_organism': 'Xanthomonas axonopodis pv. glycines',
            'severity': 'Moderate'
        },
        'symptoms': {
            'description': 'Bacterial pustule is characterized by small yellow-to-brown spots with raised centers (pustules) on leaf surfaces.',
            'visual_indicators': [
                'Small, light-colored pustules on leaf undersides',
                'Yellow halos around lesions',
                'Tattered leaf appearance in severe cases',
                'No water-soaking (unlike Bacterial Blight)'
            ],
            'affected_parts': ['Leaves'],
            'progression': 'Spreads by wind and splashing rain; favored by warm, wet weather.'
        },
        'precautions': {
            'seed_management': 'Use certified seeds.',
            'cultural_practices': [
                'Crop rotation',
                'Burying infected residue',
                'Avoid working in wet fields'
            ],
            'resistant_varieties': 'Many modern varieties have good resistance.',
            'monitoring': 'Check leaves during warm, rainy periods.'
        },
        'treatment': {
            'chemical_control': {
                'products': ['Copper-based bactericides'],
                'application_timing': 'Rarely economically justified unless infection is very early and severe.',
                'efficacy_note': 'Resistant varieties are the primary management tool.'
            },
            'cultural_control': ['Removal of infected residue'],
            'biological_control': 'Not common.',
            'prognosis': 'Generally less damaging than Bacterial Blight; manageable with resistance.'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced nutrition to maintain plant health',
                'potassium': 'Adequate K levels for general disease resistance'
            },
            'micronutrients': {
                'zinc': '5-10 kg ZnSO₄/ha for leaf tissue strength',
                'silicon': '100-150 kg Si/ha to strengthen cell walls'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Plant resistant varieties',
                'Use certified disease-free seeds',
                'Implement crop rotation',
                'Avoid overhead irrigation',
                'Maintain field sanitation'
            ],
            'early_detection': [
                'Monitor leaf undersides for pustules',
                'Check for yellow halos around lesions',
                'Look for tattered leaf appearance',
                'Scout during warm, wet weather'
            ],
            'integrated_approach': [
                'Prioritize resistant varieties as primary defense',
                'Combine cultural and biological controls',
                'Time copper applications appropriately if needed',
                'Maintain plant health through proper nutrition'
            ]
        },
        'economic_impact': 'Yield losses typically 5-15% in susceptible varieties; generally less damaging than bacterial blight',
        'environmental_conditions': {
            'favorable': 'Warm temperatures (24-30°C), high humidity, frequent rainfall',
            'unfavorable': 'Hot dry conditions, temperatures above 35°C'
        },
        'research_advances': [
            'Improved resistant varieties with multiple resistance genes',
            'Enhanced copper formulations with better efficacy',
            'Biological control agents for bacterial diseases',
            'Precision agriculture for targeted management'
        ]
    },
    
    'Brown Spot': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Septoria glycines (syn. Cercospora kikuchii)',
            'severity': 'Moderate to High'
        },
        'symptoms': {
            'description': 'Brown spot manifests as small, irregular brown lesions with dark borders, primarily affecting lower and middle canopy leaves.',
            'visual_indicators': [
                'Small dark brown to reddish-brown circular spots (2-5mm diameter)',
                'Angular lesions following leaf veins',
                'Purple to dark brown borders around spots',
                'Yellowing of surrounding leaf tissue',
                'Premature defoliation starting from lower leaves',
                'Reduced photosynthetic area leading to stunted growth'
            ],
            'affected_parts': ['Leaves (primary)', 'Pods (occasionally)', 'Seeds (seed staining)'],
            'progression': 'Begins on lower leaves during mid-season, progresses upward as disease intensifies'
        },
        'precautions': {
            'seed_management': 'Use fungicide-treated seeds; avoid planting infected seeds',
            'cultural_practices': [
                'Rotate with corn, wheat, or other non-host crops for 2-3 years',
                'Use resistant or tolerant soybean varieties',
                'Avoid excessive plant density; maintain 15-inch row spacing',
                'Ensure proper drainage to reduce leaf wetness duration',
                'Deep plow to bury crop residue where fungus overwinters'
            ],
            'resistant_varieties': 'Varieties with partial resistance show reduced disease severity',
            'monitoring': 'Begin scouting at R1 (flowering) stage; inspect lower leaves first'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Azoxystrobin (strobilurin fungicide)',
                    'Tebuconazole (triazole fungicide)',
                    'Picoxystrobin + Cyproconazole (combination product)',
                    'Pyraclostrobin + Fluxapyroxad'
                ],
                'application_timing': 'Apply at R1-R3 stages when disease is first detected',
                'efficacy_note': 'Best results when applied before 10% leaf infection'
            },
            'cultural_control': [
                'Remove and destroy infected crop residue after harvest',
                'Implement clean cultivation practices',
                'Reduce irrigation frequency during reproductive stages'
            ],
            'biological_control': 'Trichoderma species can suppress fungal growth',
            'prognosis': 'Good control achievable with integrated management; 2-3 sprays may be needed'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': '10-26-26 or 12-32-16 depending on soil test',
                'nitrogen': 'Moderate N (30-50 kg/ha); avoid late-season N',
                'phosphorus': '70-90 kg P₂O₅/ha',
                'potassium': '60-80 kg K₂O/ha - critical for disease resistance'
            },
            'micronutrients': {
                'sulfur': '20-30 kg S/ha enhances fungicidal activity',
                'manganese': '5 kg MnSO₄/ha as foliar spray',
                'zinc': '3-5 kg ZnSO₄/ha'
            },
            'organic_amendments': [
                'Well-decomposed FYM (10-15 tons/ha)',
                'Green manure with legumes before soybean',
                'Biochar (2-3 tons/ha) for soil health'
            ],
            'soil_management': 'Maintain pH 6.2-6.8; lime application if pH < 5.8'
        },
        'management_strategies': {
            'preventive_measures': [
                'Use brown spot resistant/tolerant varieties',
                'Apply fungicide seed treatments',
                'Implement 2-3 year crop rotation',
                'Maintain proper drainage',
                'Optimize plant spacing for air circulation'
            ],
            'early_detection': [
                'Scout lower leaves at R1 stage',
                'Look for small brown circular spots',
                'Monitor for angular lesions following leaf veins',
                'Check for purple borders around spots'
            ],
            'integrated_approach': [
                'Combine resistant varieties with fungicide applications',
                'Time fungicide applications at R1-R3 stages',
                'Maintain field hygiene and crop rotation',
                'Use cultural practices to reduce inoculum'
            ]
        },
        'economic_impact': 'Yield losses of 15-30% in susceptible varieties; seed quality also affected',
        'environmental_conditions': {
            'favorable': 'Temperature 25-30°C, high relative humidity (>85%), prolonged leaf wetness',
            'unfavorable': 'Low humidity (<60%), temperatures >35°C inhibit fungal growth'
        },
        'research_advances': [
            'Improved resistant varieties with multiple resistance genes',
            'New fungicide chemistries with novel modes of action',
            'Precision application technologies',
            'Remote sensing for early detection'
        ]
    },

    'Cercospora Leaf Blight': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Cercospora kikuchii',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Cercospora leaf blight causes a distinct purple-to-bronze discoloration of the upper leaves in the sun-exposed canopy.',
            'visual_indicators': [
                'Purple/bronze tint on upper leaf surfaces',
                'Leathery leaf texture',
                'Purple staining on seeds (Purple Seed Stain)',
                'Premature defoliation'
            ],
            'affected_parts': ['Upper Leaves', 'Seeds', 'Petioles'],
            'progression': 'Symptoms appear during seed-fill (R5-R6); favored by high humidity and sunlight.'
        },
        'precautions': {
            'seed_management': 'Use fungicide-treated seeds to reduce early-season infection.',
            'cultural_practices': [
                'Crop rotation',
                'Tillage to reduce overwintering inoculum'
            ],
            'monitoring': 'Scout fields at early reproductive stages.'
        },
        'treatment': {
            'chemical_control': {
                'products': ['Quinone Outside Inhibitors (QoI)', 'DMI fungicides'],
                'application_timing': 'R3-R5 stages are critical.',
                'efficacy_note': 'Fungicide resistance is common; use mixed modes of action.'
            }
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced nutrition to maintain plant health',
                'potassium': 'High K levels for disease resistance and plant health'
            },
            'micronutrients': {
                'silicon': '100-150 kg Si/ha to strengthen leaf tissues',
                'calcium': '200-300 kg Ca/ha for cell wall integrity'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Use resistant/tolerant varieties',
                'Apply fungicide seed treatments',
                'Implement crop rotation',
                'Time planting to avoid stress periods',
                'Maintain proper plant spacing'
            ],
            'early_detection': [
                'Monitor upper leaves for purple/bronze discoloration',
                'Check for leathery leaf texture',
                'Look for purple seed staining',
                'Scout during R5-R6 stages'
            ],
            'integrated_approach': [
                'Combine resistant varieties with fungicide applications',
                'Time fungicide applications at R3-R5 stages',
                'Use mixed modes of action to prevent resistance',
                'Maintain field hygiene and rotation practices'
            ]
        },
        'economic_impact': 'Yield losses of 10-30% in susceptible varieties; seed quality significantly affected by purple seed stain',
        'environmental_conditions': {
            'favorable': 'High humidity (>85%), warm temperatures (25-30°C), intense sunlight',
            'unfavorable': 'Dry conditions, temperatures above 35°C'
        },
        'research_advances': [
            'New resistant varieties with improved tolerance',
            'Advanced fungicide formulations with dual modes of action',
            'Precision agriculture for targeted applications',
            'Breeding for combined disease resistance'
        ]
    },

    'Downey Mildew': {
        'disease_type': {
            'classification': 'Fungal Disease (Oomycete)',
            'causal_organism': 'Peronospora manshurica',
            'severity': 'Low to Moderate'
        },
        'symptoms': {
            'description': 'Downey mildew appears as pale green to yellow spots on the upper leaf surface, with gray fuzzy growth on the underside.',
            'visual_indicators': [
                'Yellow-green spots on upper leaf surface',
                'Gray/purple fuzzy fungal growth on leaf undersides',
                'Systemically infected leaves are small and distorted',
                'Seeds may be covered with white fungal crust'
            ],
            'affected_parts': ['Leaves', 'Seeds'],
            'progression': 'Favored by high humidity and cool temperatures.'
        },
        'precautions': {
            'seed_management': 'Use fungicide seed treatments.',
            'cultural_practices': ['Crop rotation', 'Plowing under residue']
        },
        'treatment': {
            'chemical_control': {
                'products': ['Mefenoxam', 'Metalaxyl (seed treatment)', 'Foliar fungicides'],
                'application_timing': 'Rarely needs foliar treatment.'
            }
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced nutrition to maintain plant health',
                'phosphorus': 'Adequate P for root development and plant health',
                'potassium': 'K for overall plant resistance'
            },
            'micronutrients': {
                'manganese': '5-8 kg Mn/ha for plant immunity',
                'zinc': '5-10 kg Zn/ha for plant health'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Use high-quality seeds with fungicide treatment',
                'Implement crop rotation',
                'Improve field drainage',
                'Avoid excessive plant density',
                'Monitor weather conditions for disease favorability'
            ],
            'early_detection': [
                'Check upper leaf surfaces for yellow-green spots',
                'Examine leaf undersides for fuzzy growth',
                'Monitor for distorted leaf growth',
                'Look for white fungal crust on seeds'
            ],
            'integrated_approach': [
                'Focus on seed treatments as primary defense',
                'Improve field drainage to reduce humidity',
                'Use cultural practices to minimize disease pressure',
                'Monitor and scout regularly during cool, humid periods'
            ]
        },
        'economic_impact': 'Yield losses typically 5-15% in susceptible varieties; more common in northern growing regions',
        'environmental_conditions': {
            'favorable': 'Cool temperatures (15-22°C), high humidity (>85%), extended leaf wetness',
            'unfavorable': 'Warm temperatures (>25°C), dry conditions'
        },
        'research_advances': [
            'Improved seed treatment fungicides',
            'Resistant varieties with better tolerance',
            'Precision agriculture for targeted management',
            'Weather-based prediction models'
        ]
    },

    'Frogeye Leaf Spot': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Cercospora sojina',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Frogeye leaf spot creates small, circular-to-angular spots with gray centers and dark reddish-brown borders.',
            'visual_indicators': [
                'Circular spots with gray centers and red-brown rims',
                'Tiny black dots (conidia) in the gray center',
                'Stem and pod lesions in severe cases'
            ],
            'affected_parts': ['Leaves', 'Stems', 'Pods'],
            'progression': 'Spreads by wind-blown spores; favored by warm, humid weather.'
        },
        'precautions': {
            'resistant_varieties': 'Use varieties with Rcs3 resistance gene.',
            'cultural_practices': ['Crop rotation', 'Tillage']
        },
        'treatment': {
            'chemical_control': {
                'products': ['Strobilurins', 'Triazoles'],
                'application_timing': 'R3 stage application if disease is present.',
                'efficacy_note': 'Resistance to strobilurins is widespread in some regions.'
            }
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced fertility program',
                'potassium': 'High K levels for improved disease resistance',
                'phosphorus': 'Adequate P for plant health and immune function'
            },
            'micronutrients': {
                'silicon': '100-150 kg Si/ha to strengthen leaf tissues',
                'zinc': '5-10 kg Zn/ha for plant immunity',
                'manganese': '5 kg Mn/ha for enzyme function'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Use resistant varieties with Rcs3 gene',
                'Implement 2-3 year crop rotation',
                'Use fungicide seed treatments',
                'Time fungicide applications appropriately',
                'Monitor fields for early detection'
            ],
            'early_detection': [
                'Look for circular spots with gray centers',
                'Check for red-brown borders around lesions',
                'Monitor for tiny black dots (conidia) in centers',
                'Inspect stems and pods for lesions in severe cases'
            ],
            'integrated_approach': [
                'Combine resistant varieties with fungicide applications',
                'Rotate fungicide modes of action to prevent resistance',
                'Use crop rotation to reduce inoculum',
                'Time applications at R3 stage if needed'
            ]
        },
        'economic_impact': 'Yield losses of 10-50% in susceptible varieties under favorable conditions; significant impact on seed quality',
        'environmental_conditions': {
            'favorable': 'Warm temperatures (20-28°C), high humidity (>80%), frequent rainfall',
            'unfavorable': 'Hot dry conditions, temperatures above 32°C'
        },
        'research_advances': [
            'New resistant varieties with stacked resistance genes',
            'Advanced fungicide chemistries with multiple sites of action',
            'Precision agriculture for targeted applications',
            'Molecular diagnostics for early detection'
        ]
    },

    'Powdery Mildew': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Microsphaera diffusa',
            'severity': 'Low to Moderate'
        },
        'symptoms': {
            'description': 'Powdery mildew is easily recognized by the white, flour-like fungal growth on leaf surfaces.',
            'visual_indicators': [
                'White powdery patches on leaves, stems, and pods',
                'Premature leaf aging and drop',
                'Reduced photosynthesis'
            ]
        },
        'treatment': {
            'chemical_control': {
                'products': ['Sulfur-based fungicides', 'Triazoles'],
                'application_timing': 'Apply if disease appears before pod-fill and conditions persist.'
            }
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced nutrition to maintain plant health',
                'nitrogen': 'Moderate N levels; excessive N promotes disease',
                'potassium': 'Adequate K for plant resistance'
            },
            'micronutrients': {
                'silicon': '100-150 kg Si/ha to strengthen leaf surfaces',
                'calcium': '200-300 kg Ca/ha for cell wall strength'
            },
            'organic_amendments': [
                'Compost (5-10 tons/ha) to enhance plant health',
                'Mycorrhizal fungi inoculants'
            ]
        },
        'management_strategies': {
            'preventive_measures': [
                'Improve air circulation with proper spacing',
                'Avoid excessive nitrogen fertilization',
                'Monitor fields during warm, dry conditions',
                'Use resistant varieties when available',
                'Maintain field hygiene'
            ],
            'early_detection': [
                'Look for white powdery patches on leaf surfaces',
                'Check stems and pods for fungal growth',
                'Monitor for premature leaf aging',
                'Scout during warm, dry weather with high humidity at night'
            ],
            'integrated_approach': [
                'Combine cultural practices with fungicide applications',
                'Time sulfur applications appropriately',
                'Use triazole fungicides for systemic control',
                'Maintain plant health through proper nutrition'
            ]
        },
        'economic_impact': 'Yield losses of 5-15% in susceptible varieties; more common in stressed plants',
        'environmental_conditions': {
            'favorable': 'Warm days (21-29°C), cool nights, high humidity at night, dry conditions',
            'unfavorable': 'Consistent rainfall, extreme heat, poor air circulation'
        },
        'research_advances': [
            'Resistant varieties with improved tolerance',
            'Biological control agents targeting powdery mildew',
            'Advanced sulfur formulations with extended residual',
            'Environmental monitoring systems for prediction'
        ]
    },

    'Target Spot': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Corynespora cassiicola',
            'severity': 'Moderate'
        },
        'symptoms': {
            'description': 'Target spot lesions are circular with alternating light and dark brown rings, resembling a target.',
            'visual_indicators': [
                'Concentric rings in brown leaf spots',
                'Yellow halos around spots',
                'Premature defoliation starting in the lower canopy'
            ]
        },
        'treatment': {
            'chemical_control': {
                'products': ['Strobilurins', 'SDHI fungicides'],
                'application_timing': 'R1-R3 stages if conditions are favorable (high humidity).'
            }
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced fertility program',
                'potassium': 'High K levels for disease resistance',
                'phosphorus': 'Adequate P for root and plant health'
            },
            'micronutrients': {
                'silicon': '100-150 kg Si/ha to strengthen leaf tissues',
                'zinc': '5-10 kg Zn/ha for plant immunity'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Implement crop rotation with non-host crops',
                'Use resistant varieties when available',
                'Maintain proper plant spacing for air circulation',
                'Monitor weather conditions for disease favorability',
                'Remove crop debris after harvest'
            ],
            'early_detection': [
                'Look for concentric rings in leaf spots',
                'Check for yellow halos around lesions',
                'Monitor lower canopy for premature defoliation',
                'Scout during high humidity conditions'
            ],
            'integrated_approach': [
                'Combine cultural practices with fungicide applications',
                'Time fungicide applications at R1-R3 stages',
                'Use SDHI fungicides for effective control',
                'Maintain field hygiene to reduce inoculum'
            ]
        },
        'economic_impact': 'Yield losses of 10-20% in susceptible varieties under favorable conditions',
        'environmental_conditions': {
            'favorable': 'Warm temperatures (22-28°C), high humidity (>85%), prolonged leaf wetness',
            'unfavorable': 'Hot dry conditions, temperatures above 32°C'
        },
        'research_advances': [
            'Improved resistant varieties with multiple resistance genes',
            'Advanced fungicide chemistries with novel modes of action',
            'Precision agriculture for targeted management',
            'Molecular tools for pathogen identification'
        ]
    },
    
    'Insects': {
        'disease_type': {
            'classification': 'Pest Infestation',
            'causal_organism': 'Various (Aphids, Beetles, Stink bugs, etc.)',
            'severity': 'Variable'
        },
        'symptoms': {
            'description': 'Damage varies by insect type, including leaf defoliation, sucking of sap, or pod damage.',
            'visual_indicators': [
                'Chewed leaf edges or holes',
                'Yellowing/curling (aphid damage)',
                'Shriveled seeds (stink bug damage)',
                'Visible pests on stems or leaf undersides'
            ],
            'affected_parts': ['Leaves', 'Pods', 'Stems'],
            'progression': 'Varies by pest; can spread rapidly in warm weather.'
        },
        'precautions': {
            'seed_management': 'N/A',
            'cultural_practices': [
                'Regular scouting',
                'Encourage beneficial insects',
                'Use trap crops'
            ],
            'monitoring': 'Check fields weekly; use economic thresholds for spraying.'
        },
        'treatment': {
            'chemical_control': {
                'products': ['Pyrethroids', 'Neonicotinoids', 'Organophosphates'],
                'application_timing': 'Apply when pest numbers exceed economic thresholds.'
            },
            'biological_control': 'Release of ladybugs, lacewings, or parasitic wasps.',
            'prognosis': 'Generally manageable with timely monitoring and intervention.'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced nutrition to maintain plant health and resilience',
                'potassium': 'Adequate K for plant defense mechanisms',
                'phosphorus': 'Sufficient P for root development and plant vigor'
            },
            'micronutrients': {
                'silicon': '100-150 kg Si/ha to strengthen plant tissues',
                'boron': '0.5-1.0 kg B/ha for cell wall strength'
            }
        },
        'management_strategies': {
            'preventive_measures': [
                'Regular field scouting and monitoring',
                'Encourage beneficial insect populations',
                'Use trap crops to monitor pest movement',
                'Plant at optimal timing to avoid peak pest pressure',
                'Maintain field hygiene and remove alternative hosts'
            ],
            'early_detection': [
                'Monitor for visible pest presence',
                'Check for feeding damage on leaves and pods',
                'Look for yellowing or curling leaves',
                'Scout for shriveled seeds and damaged pods'
            ],
            'integrated_approach': [
                'Combine biological, cultural, and chemical controls',
                'Use economic thresholds for treatment decisions',
                'Rotate insecticide modes of action to prevent resistance',
                'Preserve beneficial insect populations'
            ]
        },
        'economic_impact': 'Yield losses vary widely (5-40%) depending on pest type, population density, and timing of infestation',
        'environmental_conditions': {
            'favorable': 'Warm temperatures (20-30°C), moderate humidity, absence of natural enemies',
            'unfavorable': 'Cold temperatures, heavy rainfall, presence of beneficial insects'
        },
        'research_advances': [
            'Biotechnology for pest-resistant varieties',
            'Precision agriculture for targeted insecticide applications',
            'Integrated pest management programs',
            'Biological control enhancement techniques'
        ]
    },
    
    'Nutrient Deficiencies': {
        'disease_type': {
            'classification': 'Physiological Disorder / Nutrient Deficiency',
            'causal_organism': 'Deficiency of essential nutrients (Nitrogen, Phosphorus, Potassium, Iron, etc.)',
            'severity': 'Moderate'
        },
        'symptoms': {
            'description': 'Nutrient deficiencies present as distinctive yellowing, stunting, or discoloration of leaves depending on the missing element.',
            'visual_indicators': [
                'Interveinal chlorosis (yellowing between veins)',
                'Young leaves most severely affected',
                'Leaf veins remain dark green creating striking contrast',
                'Stunted plant growth and reduced vigor',
                'Poor root development',
                'Reduced chlorophyll formation',
                'In severe cases, leaf edges turn brown and necrotic'
            ],
            'affected_parts': ['Young leaves (primary)', 'Growing points', 'Entire plant (systemic)'],
            'progression': 'Starts in youngest leaves at growing tips; spreads to older leaves if uncorrected'
        },
        'precautions': {
            'seed_management': 'Use quality seeds with high vigor',
            'cultural_practices': [
                'Maintain optimal soil pH (6.0-6.8); avoid alkaline conditions',
                'Improve soil drainage in waterlogged areas',
                'Avoid excessive phosphorus application which interferes with iron uptake',
                'Reduce bicarbonate levels in irrigation water',
                'Incorporate organic matter to improve iron availability',
                'Plant iron-efficient soybean varieties'
            ],
            'resistant_varieties': 'Varieties with "Fe" designation show improved iron efficiency',
            'monitoring': 'Monitor soil pH and conduct tissue analysis if chlorosis appears'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Iron chelate (Fe-EDDHA) for alkaline soils',
                    'Iron sulfate (FeSO₄) for acidic soils',
                    'Chelated iron foliar sprays (Fe-EDTA)',
                    'Iron-containing liquid fertilizers'
                ],
                'application_timing': 'Foliar spray at first sign of chlorosis; repeat every 7-10 days',
                'efficacy_note': 'Foliar application provides quick but temporary relief; soil treatment needed for long-term correction'
            },
            'cultural_control': [
                'Apply sulfur (100-200 kg/ha) to reduce soil pH',
                'Improve drainage with raised beds or tile drainage',
                'Reduce phosphorus application if soil P is high',
                'Apply organic acids to increase iron solubility'
            ],
            'biological_control': 'Not applicable (non-pathogenic condition)',
            'prognosis': 'Excellent recovery with proper iron supplementation and pH management'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Balanced 15-15-15; avoid excess phosphorus',
                'nitrogen': '40-50 kg/ha as urea or ammonium sulfate',
                'phosphorus': 'Moderate levels (40-60 kg P₂O₅/ha); high P inhibits Fe uptake',
                'potassium': '50-60 kg K₂O/ha'
            },
            'micronutrients': {
                'iron': '10-20 kg Fe/ha as soil application; 0.5-1% as foliar spray',
                'zinc': '5 kg Zn/ha (Fe and Zn work synergistically)',
                'manganese': '2-3 kg Mn/ha',
                'sulfur': '20-30 kg S/ha to acidify soil'
            },
            'organic_amendments': [
                'Compost (10-15 tons/ha) - improves Fe availability',
                'FYM (15-20 tons/ha)',
                'Green manure crops to add organic matter',
                'Sulfur-enriched compost'
            ],
            'soil_management': 'Critical: Maintain pH 6.0-6.8; test soil pH regularly; lime only when pH <5.5'
        },
        'economic_impact': 'Yield reduction 10-25% in moderate deficiency; severe deficiency can cause 30-50% loss',
        'environmental_conditions': {
            'favorable': 'Alkaline soils (pH >7.5), waterlogged conditions, cold soil temperatures',
            'unfavorable': 'Well-drained, slightly acidic soils with adequate organic matter'
        }
    },
    
    'Healthy': {
        'disease_type': {
            'classification': 'Healthy Plant (No Disease)',
            'causal_organism': 'None',
            'severity': 'None'
        },
        'symptoms': {
            'description': 'Healthy soybean plants exhibit vigorous growth, uniform green coloration, and normal developmental progression.',
            'visual_indicators': [
                'Uniform dark green leaf color throughout canopy',
                'Strong, erect stems with no discoloration',
                'Robust root system with active nodulation',
                'Normal flower formation and pod set',
                'Leaves free from spots, lesions, or discoloration',
                'Appropriate growth rate for variety and growth stage',
                'No signs of wilting, chlorosis, or necrosis',
                'Healthy trifoliate leaf structure'
            ],
            'affected_parts': 'None - all plant parts healthy',
            'progression': 'Normal developmental progression: VE→VC→V1-V6→R1-R8'
        },
        'precautions': {
            'seed_management': 'Continue using certified seeds; maintain seed quality standards',
            'cultural_practices': [
                'Maintain consistent irrigation schedule (1-1.5 inches/week)',
                'Continue balanced fertilization program',
                'Regular field monitoring (weekly) for early disease/pest detection',
                'Maintain proper plant density and row spacing',
                'Ensure good field drainage',
                'Practice crop rotation to prevent disease buildup',
                'Remove weed competition promptly'
            ],
            'resistant_varieties': 'Continue with proven varieties adapted to your region',
            'monitoring': 'Continue routine scouting even when plants appear healthy'
        },
        'treatment': {
            'chemical_control': {
                'products': 'No pesticides needed; avoid unnecessary chemical applications',
                'application_timing': 'N/A',
                'efficacy_note': 'Preventive fungicide may be considered if disease pressure expected'
            },
            'cultural_control': [
                'Maintain current management practices',
                'Continue monitoring for any changes',
                'Ensure optimal growing conditions'
            ],
            'biological_control': 'Encourage beneficial insects and soil microbes',
            'prognosis': 'Excellent - continue good agricultural practices'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Based on soil test results; typical 20-40-40 kg/ha',
                'nitrogen': '20-30 kg N/ha (soybean fixes most N from atmosphere)',
                'phosphorus': '60-80 kg P₂O₅/ha for optimal nodulation',
                'potassium': '40-60 kg K₂O/ha for pod filling'
            },
            'micronutrients': {
                'sulfur': '10-20 kg S/ha for protein synthesis',
                'boron': '0.5-1.0 kg B/ha for flower and pod retention',
                'zinc': '5 kg Zn/ha',
                'molybdenum': '25-50 g Mo/ha for N-fixation'
            },
            'organic_amendments': [
                'Well-decomposed compost (5-8 tons/ha)',
                'Vermicompost (2-3 tons/ha)',
                'Green manure incorporation',
                'Rhizobium inoculant at planting'
            ],
            'soil_management': 'Maintain pH 6.0-7.0; add lime if pH <6.0; sulfur if pH >7.5'
        },
        'economic_impact': 'Maximum yield potential achievable with optimal conditions',
        'environmental_conditions': {
            'favorable': 'Current conditions are optimal - maintain them',
            'unfavorable': 'Monitor weather forecasts for potential stress conditions'
        }
    },
    
    'Mosaic Virus': {
        'disease_type': {
            'classification': 'Viral Disease',
            'causal_organism': 'Soybean mosaic virus (SMV) - Potyvirus family',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Soybean mosaic virus causes distinctive mottled patterns of light and dark green areas on leaves, with variable severity.',
            'visual_indicators': [
                'Mosaic or mottled pattern of yellow-green and dark green on leaves',
                'Leaf puckering and distortion',
                'Stunted plant growth (10-30% height reduction)',
                'Delayed maturity',
                'Downward curling of leaf margins',
                'Seed mottling and reduced seed quality',
                'Reduced pod numbers',
                'Symptoms more severe in younger plants'
            ],
            'affected_parts': ['Leaves (primary)', 'Seeds', 'Entire plant (systemic)'],
            'progression': 'Virus is systemic; spreads from infection point throughout plant; symptoms most visible on young leaves'
        },
        'precautions': {
            'seed_management': 'Use certified virus-free seeds - critical for prevention',
            'cultural_practices': [
                'Plant resistant varieties with Rsv genes (Rsv1, Rsv3, Rsv4)',
                'Control aphid vectors with systemic insecticides at planting',
                'Remove infected plants immediately to prevent spread',
                'Eliminate volunteer soybean and weed hosts (alternative hosts)',
                'Avoid planting near infected fields',
                'Delay planting to reduce early-season aphid exposure',
                'Use reflective mulches to repel aphids'
            ],
            'resistant_varieties': 'Varieties with Rsv resistance genes provide excellent protection',
            'monitoring': 'Scout fields for aphids and virus symptoms from emergence through R2'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Imidacloprid (seed treatment for aphid control)',
                    'Thiamethoxam (systemic aphid control)',
                    'Acetamiprid (aphid vector control)',
                    'Mineral oil (interferes with virus transmission)'
                ],
                'application_timing': 'Apply at planting and early vegetative stages for aphid control',
                'efficacy_note': 'No chemical can cure viral infection; focus is on vector control'
            },
            'cultural_control': [
                'Rogue (remove) infected plants immediately',
                'Destroy infected plants by burning or deep burying',
                'Sanitize equipment between fields',
                'Plant virus-free seeds only'
            ],
            'biological_control': 'Encourage aphid predators: ladybugs, lacewings, parasitic wasps',
            'prognosis': 'No cure once infected; focus on prevention and limiting spread'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': 'Moderate levels 15-30-30 kg/ha',
                'nitrogen': 'Moderate N (30-40 kg/ha); excess promotes aphid reproduction',
                'phosphorus': '60-80 kg P₂O₅/ha',
                'potassium': '60-80 kg K₂O/ha - strengthens plant immunity'
            },
            'micronutrients': {
                'zinc': '5-8 kg Zn/ha to boost immunity',
                'boron': '0.5-1.0 kg B/ha',
                'silicon': '100 kg Si/ha - enhances resistance'
            },
            'organic_amendments': [
                'Compost (8-10 tons/ha) for plant health',
                'Neem cake (200 kg/ha) - repels aphids',
                'Vermicompost (3-4 tons/ha)',
                'Seaweed extract as foliar spray - improves stress tolerance'
            ],
            'soil_management': 'Maintain optimal pH 6.2-7.0; healthy vigorous plants better tolerate virus'
        },
        'management_strategies': {
            'preventive_measures': [
                'Use virus-resistant varieties with Rsv genes',
                'Plant certified virus-free seeds',
                'Control aphid vectors with seed treatments',
                'Eliminate volunteer soybeans and alternative hosts',
                'Time planting to avoid peak aphid periods'
            ],
            'early_detection': [
                'Monitor for mottled leaf patterns',
                'Check for leaf puckering and distortion',
                'Look for stunted plant growth',
                'Scout for aphid populations'
            ],
            'integrated_approach': [
                'Combine resistant varieties with vector control',
                'Use reflective mulches to repel aphids',
                'Maintain field sanitation',
                'Time insecticide applications appropriately'
            ]
        },
        'economic_impact': 'Yield losses 20-50% in susceptible varieties; seed quality significantly reduced',
        'environmental_conditions': {
            'favorable': 'Moderate temperatures (20-28°C) favor aphid activity and virus transmission',
            'unfavorable': 'Hot weather (>35°C) reduces aphid populations'
        },
        'research_advances': [
            'Stacked resistance genes for durable protection',
            'RNAi-based virus resistance',
            'Advanced vector management techniques',
            'Molecular diagnostics for early detection'
        ]
    },
    
    'Rust': {
        'disease_type': {
            'classification': 'Fungal Disease',
            'causal_organism': 'Phakopsora pachyrhizi (Asian soybean rust)',
            'severity': 'Very High - most destructive soybean disease'
        },
        'symptoms': {
            'description': 'Soybean rust is characterized by small tan to reddish-brown pustules on leaf surfaces, capable of causing severe defoliation.',
            'visual_indicators': [
                'Small (0.2-0.5mm) tan, brown, or reddish pustules on lower leaf surface',
                'Pustules release orange-brown spores when rubbed',
                'Yellow halos around pustules on upper leaf surface',
                'Premature leaf yellowing and drop',
                'Severe defoliation starting from lower canopy',
                'Reduced pod fill and seed weight',
                'Can defoliate entire plant in severe outbreaks'
            ],
            'affected_parts': ['Leaves (primary - lower surface)', 'Petioles', 'Pods (occasionally)'],
            'progression': 'Starts on lower leaves; rapidly moves up canopy; can defoliate field in 2-3 weeks if unchecked'
        },
        'precautions': {
            'seed_management': 'Use high-quality seeds; early planting reduces disease exposure',
            'cultural_practices': [
                'Plant early-maturing varieties to escape late-season disease',
                'Use resistant varieties where available (partial resistance)',
                'Avoid dense plantings - maintain 15-30 inch row spacing',
                'Monitor disease spread using sentinel plots',
                'Remove volunteer soybean plants (green bridge)',
                'Practice good weed management',
                'Deep plow crop residue'
            ],
            'resistant_varieties': 'Few resistant varieties; use partial resistance and fungicides',
            'monitoring': 'Scout fields twice weekly from R1 onwards; check lower leaf surfaces'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Azoxystrobin + Cyproconazole (highly effective)',
                    'Picoxystrobin + Cyproconazole',
                    'Trifloxystrobin + Prothioconazole',
                    'Tebuconazole (triazole)',
                    'Pyraclostrobin + Epoxiconazole'
                ],
                'application_timing': 'Apply preventively at R1-R2 or at first detection; second application 14-21 days later',
                'efficacy_note': 'Early preventive application essential; disease spreads rapidly. Use strobilurin + triazole combinations'
            },
            'cultural_control': [
                'Early harvest if disease detected late-season',
                'Destroy crop residue after harvest',
                'Avoid late planting'
            ],
            'biological_control': 'Limited options; some Trichoderma species show promise',
            'prognosis': 'Good control with timely fungicide application; delay of even 1 week can result in significant yield loss'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': '15-30-30 or as per soil test',
                'nitrogen': 'Moderate N (40-50 kg/ha); high N delays maturity and prolongs exposure',
                'phosphorus': '70-90 kg P₂O₅/ha',
                'potassium': '70-90 kg K₂O/ha - critical for disease resistance'
            },
            'micronutrients': {
                'sulfur': '30-40 kg S/ha - enhances fungicide efficacy',
                'silicon': '100-150 kg Si/ha - strengthens cell walls',
                'manganese': '5 kg Mn/ha',
                'zinc': '5 kg Zn/ha'
            },
            'organic_amendments': [
                'Compost (10 tons/ha)',
                'FYM (12-15 tons/ha)',
                'Potassium-rich organic materials'
            ],
            'soil_management': 'Maintain pH 6.0-6.5; ensure adequate drainage'
        },
        'management_strategies': {
            'preventive_measures': [
                'Use rust-resistant varieties with available partial resistance',
                'Monitor sentinel plots for early detection',
                'Remove volunteer soybeans that serve as inoculum source',
                'Plan fungicide applications before disease establishment',
                'Coordinate with regional disease monitoring programs'
            ],
            'early_detection': [
                'Check lower leaf surfaces for small pustules',
                'Look for orange-brown spore masses when rubbed',
                'Monitor for yellow halos on upper leaf surface',
                'Scout twice weekly during R1 and beyond'
            ],
            'integrated_approach': [
                'Combine resistant varieties with preventive fungicide programs',
                'Use strobilurin + triazole combinations for best efficacy',
                'Apply fungicides at first detection or preventively',
                'Coordinate with regional monitoring to time applications'
            ]
        },
        'economic_impact': 'Catastrophic losses possible: 30-80% yield reduction if untreated; economic losses exceed $2 billion annually in affected regions',
        'environmental_conditions': {
            'favorable': 'Temperature 20-28°C, >6 hours leaf wetness, high humidity (>75%), frequent dew or light rain',
            'unfavorable': 'Hot dry conditions (>35°C), low humidity (<60%)'
        },
        'research_advances': [
            'New resistant varieties with multiple resistance genes',
            'Advanced fungicide chemistries with dual modes of action',
            'Real-time disease prediction models',
            'Remote sensing for early detection and mapping'
        ]
    },
    
    'Southern Blight': {
        'disease_type': {
            'classification': 'Fungal Disease (Soil-borne)',
            'causal_organism': 'Sclerotium rolfsii (also called Athelia rolfsii)',
            'severity': 'High in affected areas'
        },
        'symptoms': {
            'description': 'Southern blight is a devastating soil-borne disease causing rapid wilt and death with characteristic white fungal growth at the soil line.',
            'visual_indicators': [
                'Sudden wilting of entire plant',
                'Water-soaked lesions on stem near soil line',
                'White cottony fungal growth (mycelium) at stem base',
                'Tan to brown, round sclerotia (1-2mm) resembling mustard seeds',
                'Stem rot and plant collapse',
                'Yellowing and wilting starting from lower leaves',
                'Plant death within days of symptom appearance',
                'Often occurs in patches or circular patterns in field'
            ],
            'affected_parts': ['Stem base', 'Roots', 'Lower pods', 'Entire plant (wilting)'],
            'progression': 'Appears mid-to-late season during hot weather; spreads in circular patterns from infection centers'
        },
        'precautions': {
            'seed_management': 'Use disease-free seeds; seed treatment helps but has limited effect',
            'cultural_practices': [
                'Deep plow (20-25cm) to bury sclerotia deep in soil',
                'Rotate with non-susceptible crops: corn, sorghum, small grains (3-4 year rotation)',
                'Avoid dense planting; use wider row spacing (30 inches)',
                'Improve soil drainage with raised beds or land leveling',
                'Reduce soil contact with lower pods through mulching',
                'Solarize soil in hotspots during summer fallow',
                'Remove and destroy infected plants',
                'Avoid excess nitrogen and over-irrigation'
            ],
            'resistant_varieties': 'Limited genetic resistance available; focus on cultural management',
            'monitoring': 'Scout fields during hot humid weather; check stem bases for white growth'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Flutolanil (soil application)',
                    'Azoxystrobin (seed treatment + foliar)',
                    'Tebuconazole (soil drench)',
                    'PCNB (Pentachloronitrobenzene) - soil incorporation',
                    'Thiram (seed treatment)'
                ],
                'application_timing': 'Preventive soil treatment before planting; stem-directed spray at R1',
                'efficacy_note': 'Chemical control difficult; best results with preventive soil application + cultural practices'
            },
            'cultural_control': [
                'Remove infected plants with surrounding soil',
                'Solarize small infected areas',
                'Improve field drainage',
                'Avoid moving soil from infected to healthy areas'
            ],
            'biological_control': [
                'Trichoderma harzianum (fungal antagonist)',
                'Bacillus subtilis',
                'Gliocladium virens',
                'Organic amendments promote antagonistic microbes'
            ],
            'prognosis': 'Difficult to control once established; integrated management essential'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': '12-24-24 kg/ha or as per soil test',
                'nitrogen': 'Low to moderate N (30-40 kg/ha); excess N increases susceptibility',
                'phosphorus': '50-70 kg P₂O₅/ha',
                'potassium': '50-70 kg K₂O/ha'
            },
            'micronutrients': {
                'calcium': '200-300 kg Ca/ha - strengthens cell walls and inhibits fungus',
                'sulfur': '25-35 kg S/ha',
                'boron': '1.0 kg B/ha'
            },
            'organic_amendments': [
                'Well-composted FYM (12-15 tons/ha) - increases antagonistic microbes',
                'Mustard or neem cake (300-400 kg/ha)',
                'Biochar (2-3 tons/ha) for soil health',
                'Calcium-rich amendments (gypsum, lime)'
            ],
            'soil_management': 'Maintain pH 6.0-6.5; add calcium amendments; increase organic matter to suppress disease'
        },
        'management_strategies': {
            'preventive_measures': [
                'Deep plow to bury sclerotia deeply',
                'Implement 3-4 year crop rotation with non-hosts',
                'Use wide row spacing for better air circulation',
                'Improve soil drainage systems',
                'Solarize soil in infected hotspots'
            ],
            'early_detection': [
                'Monitor stem bases for white fungal growth',
                'Check for water-soaked lesions near soil line',
                'Look for sudden wilting of plants',
                'Identify circular patterns of infection'
            ],
            'integrated_approach': [
                'Combine deep tillage with crop rotation',
                'Use biological control agents in soil',
                'Apply preventive soil fungicides',
                'Maintain field hygiene to prevent spread'
            ]
        },
        'economic_impact': 'Severe losses (40-100%) in infected patches; disease persists in soil for years through sclerotia',
        'environmental_conditions': {
            'favorable': 'Hot temperatures (28-35°C), high soil moisture, acidic soils, high organic matter',
            'unfavorable': 'Cool temperatures (<20°C), well-drained soils, alkaline pH'
        },
        'research_advances': [
            'Biological control agents with improved efficacy',
            'Soil solarization techniques for management',
            'Marker-assisted breeding for resistance',
            'Novel fungicide chemistries for soil application'
        ]
    },
    
    'Sudden Death Syndrome': {
        'disease_type': {
            'classification': 'Fungal Disease (Soil-borne)',
            'causal_organism': 'Fusarium virguliforme (syn. F. solani f. sp. glycines)',
            'severity': 'Very High'
        },
        'symptoms': {
            'description': 'Sudden Death Syndrome causes distinctive foliar symptoms combined with root rot, appearing mid-to-late season.',
            'visual_indicators': [
                'Interveinal chlorosis (yellowing between veins) on upper leaves',
                'Yellow areas turn brown (necrotic) but veins remain green',
                'Premature defoliation; petioles remain attached',
                'Stunted growth and wilting',
                'Root system shows brown to gray discoloration',
                'Stem shows brown streaking in vascular tissue',
                'Symptoms typically appear at R3-R5 stages',
                'White to light blue fungal growth on roots under moist conditions'
            ],
            'affected_parts': ['Roots (primary infection)', 'Leaves (foliar symptoms)', 'Vascular system'],
            'progression': 'Root infection occurs early; foliar symptoms appear mid-to-late season; rapid progression once symptoms visible'
        },
        'precautions': {
            'seed_management': 'Use high-quality seeds with fungicide seed treatment',
            'cultural_practices': [
                'Plant resistant varieties (most important control measure)',
                'Delay planting until soil warms to >60°F (15.5°C)',
                'Avoid planting in poorly drained or compacted soils',
                'Implement long crop rotations (3+ years with non-hosts)',
                'Reduce soil compaction through deep tillage',
                'Avoid fields with SDS history if possible',
                'Reduce early-season stress on plants',
                'Manage soybean cyst nematode (SCN) - increases SDS severity',
                'Avoid over-irrigation in early season'
            ],
            'resistant_varieties': 'Use varieties with high SDS resistance ratings; most effective control',
            'monitoring': 'Scout for early symptoms at R3; check roots for discoloration'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Fluopyram (seed treatment - ILeVO brand)',
                    'Prothioconazole + Fluoxastrobin + Metalaxyl (seed treatment)',
                    'Mefenoxam (soil application for root pathogens)',
                    'Azoxystrobin (limited efficacy)'
                ],
                'application_timing': 'Seed treatment most effective; in-furrow application at planting',
                'efficacy_note': 'No foliar fungicides effective once symptoms appear; focus on seed treatment and prevention'
            },
            'cultural_control': [
                'Tile drainage or surface drainage improvement',
                'Reduce soil compaction',
                'Manage SCN populations through rotation and resistant varieties',
                'Avoid early planting in cold soils'
            ],
            'biological_control': [
                'Trichoderma virens (seed treatment)',
                'Bacillus firmus (nematode control)',
                'Promote beneficial soil microbes through organic amendments'
            ],
            'prognosis': 'No cure once symptoms appear; management through resistance and prevention'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': '15-30-30 kg/ha based on soil test',
                'nitrogen': '30-40 kg/ha; balanced nutrition reduces stress',
                'phosphorus': '70-90 kg P₂O₅/ha',
                'potassium': '70-90 kg K₂O/ha - high K reduces disease severity'
            },
            'micronutrients': {
                'iron': '5-10 kg Fe/ha (foliar spray if deficiency)',
                'manganese': '3-5 kg Mn/ha',
                'zinc': '5 kg Zn/ha',
                'boron': '0.5-1.0 kg B/ha'
            },
            'organic_amendments': [
                'Compost (10-12 tons/ha) to promote soil health',
                'FYM (15-18 tons/ha)',
                'Green manure to improve soil structure',
                'Balanced micronutrient applications'
            ],
            'soil_management': 'Maintain pH 6.5-7.0; optimize drainage; improve soil structure; high K levels reduce severity'
        },
        'management_strategies': {
            'preventive_measures': [
                'Use high SDS-resistant varieties',
                'Delay planting until soil warms adequately',
                'Improve field drainage and reduce compaction',
                'Manage soybean cyst nematode populations',
                'Implement long crop rotations'
            ],
            'early_detection': [
                'Check roots for brown to gray discoloration',
                'Monitor for interveinal chlorosis on upper leaves',
                'Look for brown streaking in stem vascular tissue',
                'Scout at R3-R5 stages for early symptoms'
            ],
            'integrated_approach': [
                'Combine resistant varieties with seed treatments',
                'Improve soil drainage and structure',
                'Manage SCN to reduce SDS severity',
                'Time planting to reduce stress and infection'
            ]
        },
        'economic_impact': 'Yield losses 20-80% in susceptible varieties; can devastate entire fields; damage worse with concurrent SCN infestation',
        'environmental_conditions': {
            'favorable': 'Cool wet soil early season (15-20°C) promotes root infection; warm weather later triggers foliar symptoms',
            'unfavorable': 'Well-drained soils, delayed planting, warm early season'
        },
        'research_advances': [
            'New resistant varieties with stacked resistance genes',
            'Advanced seed treatment fungicides with improved efficacy',
            'Molecular markers for early detection',
            'Integrated SCN and SDS management programs'
        ]
    },
    
    'Yellow Mosaic': {
        'disease_type': {
            'classification': 'Viral Disease',
            'causal_organism': 'Mungbean yellow mosaic virus (MYMV) or Soybean yellow mosaic virus - Begomovirus family',
            'severity': 'High'
        },
        'symptoms': {
            'description': 'Yellow mosaic causes striking yellow mottling and mosaic patterns on leaves, severely impacting plant growth and yield.',
            'visual_indicators': [
                'Bright yellow mosaic patterns alternating with green tissue',
                'Yellow mottling more prominent on younger leaves',
                'Severe stunting (plants 30-50% smaller)',
                'Leaf size reduction and distortion',
                'Reduced or absent flowering',
                'Poor pod formation and fill',
                'Entire plant may appear yellowish',
                'Symptoms appear 2-3 weeks after infection'
            ],
            'affected_parts': ['Leaves (primary)', 'Entire plant (systemic)', 'Reproductive structures'],
            'progression': 'Transmitted by whiteflies; symptoms progress from lower to upper leaves; severity increases with early infection'
        },
        'precautions': {
            'seed_management': 'Use high-quality certified seeds; not seed-transmitted but quality matters',
            'cultural_practices': [
                'Plant resistant or tolerant varieties (most important)',
                'Control whitefly populations from seedling stage',
                'Remove infected plants immediately',
                'Eliminate alternate hosts and weeds',
                'Use yellow sticky traps for monitoring whiteflies',
                'Avoid planting near legume crops with virus',
                'Adjust planting dates to avoid peak whitefly populations',
                'Use insect-proof nets in nurseries',
                'Maintain field sanitation'
            ],
            'resistant_varieties': 'Several varieties with resistance to yellow mosaic viruses available',
            'monitoring': 'Weekly monitoring for whiteflies and virus symptoms from emergence'
        },
        'treatment': {
            'chemical_control': {
                'products': [
                    'Imidacloprid (systemic - seed treatment)',
                    'Thiamethoxam (neonicotinoid)',
                    'Acetamiprid (whitefly control)',
                    'Spiromesifen (growth regulator for whiteflies)',
                    'Pymetrozine (selective whitefly control)',
                    'Buprofezin (insect growth regulator)'
                ],
                'application_timing': 'Seed treatment + foliar sprays starting at V2; spray every 7-10 days if whiteflies present',
                'efficacy_note': 'Cannot cure viral infection; intensive whitefly control required; rotate insecticide classes'
            },
            'cultural_control': [
                'Rogue infected plants weekly',
                'Use reflective mulches (silver/aluminum)',
                'Barrier crops around field perimeter',
                'Destroy infected crop residue'
            ],
            'biological_control': [
                'Chrysoperla carnea (lacewing) - whitefly predator',
                'Encarsia formosa (parasitic wasp)',
                'Beauveria bassiana (entomopathogenic fungus)',
                'Neem oil (disrupts whitefly development)'
            ],
            'prognosis': 'No cure; focus on prevention through resistant varieties and vector control'
        },
        'fertilizer_recommendations': {
            'macronutrients': {
                'NPK': '15-35-40 kg/ha',
                'nitrogen': 'Moderate N (30-40 kg/ha); excess N increases whitefly attraction',
                'phosphorus': '70-90 kg P₂O₅/ha',
                'potassium': '80-100 kg K₂O/ha - high K improves virus tolerance'
            },
            'micronutrients': {
                'zinc': '8-10 kg Zn/ha - critical for immunity',
                'boron': '1.0-1.5 kg B/ha',
                'manganese': '5 kg Mn/ha',
                'silicon': '100-150 kg Si/ha - strengthens against insects'
            },
            'organic_amendments': [
                'Neem cake (300-400 kg/ha) - repels whiteflies',
                'Compost (10-15 tons/ha)',
                'Vermicompost (4-5 tons/ha)',
                'Potassium-rich organic materials'
            ],
            'soil_management': 'Maintain pH 6.5-7.0; balanced nutrition critical for stress tolerance'
        },
        'management_strategies': {
            'preventive_measures': [
                'Use yellow mosaic-resistant varieties',
                'Control whitefly populations with seed treatments',
                'Remove alternate hosts and infected plants',
                'Use yellow sticky traps for monitoring',
                'Adjust planting dates to avoid peak whitefly periods'
            ],
            'early_detection': [
                'Monitor for bright yellow mosaic patterns on leaves',
                'Check for severe stunting of plants',
                'Look for leaf size reduction and distortion',
                'Scout for whitefly populations weekly'
            ],
            'integrated_approach': [
                'Combine resistant varieties with intensive vector control',
                'Use reflective mulches to repel whiteflies',
                'Rotate insecticide modes of action',
                'Implement cultural practices to minimize disease pressure'
            ]
        },
        'economic_impact': 'Devastating losses 40-100% if infected early; late infections cause 20-40% yield loss; severely affects seed quality',
        'environmental_conditions': {
            'favorable': 'Hot dry weather (28-35°C) favors whitefly multiplication; low humidity',
            'unfavorable': 'Heavy rainfall, cool temperatures (<20°C) reduce whitefly activity'
        },
        'research_advances': [
            'Biotechnology for virus-resistant varieties',
            'RNAi-based virus resistance',
            'Advanced vector management techniques',
            'Molecular diagnostics for early detection'
        ]
    }
}

# Metadata for dataset
DATASET_METADATA = {
    'version': '3.0',
    'last_updated': '2026-01-22',
    'total_diseases': 17,
    'data_sources': [
        'University extension services',
        'USDA agricultural research',
        'Peer-reviewed agricultural journals',
        'Field experience and agronomic practices'
    ],
    'intended_use': [
        'AI-powered disease detection systems',
        'LLM training and fine-tuning',
        'Farmer education and guidance',
        'Automated diagnostic reports',
        'RAG (Retrieval-Augmented Generation) knowledge base'
    ],
    'coverage': 'Comprehensive coverage of bacterial, fungal, viral, pest, and physiological disorders in soybean',
    'enhancements': [
        'Added management_strategies section to all diseases',
        'Added research_advances section to all diseases',
        'Expanded fertilizer_recommendations with detailed nutrients',
        'Improved point-wise formatting for better readability',
        'Enhanced environmental conditions and economic impact data'
    ]
}
