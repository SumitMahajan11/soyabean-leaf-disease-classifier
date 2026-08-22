/**
 * I18N Manager - Multi-Language Support
 * Accessibility Layer for non-English speaking farmers
 */

class I18NManager {
  constructor() {
    this.currentLang = localStorage.getItem('preferred_language') || 'en';
    this.translations = this.loadTranslations();
    console.log('🌐 I18N Manager initialized');
  }

  loadTranslations() {
    return {
      en: {
        name: 'English',
        flag: '🇬🇧',
        ui: {
          title: 'Soyabean Leaf Detection System',
          analyze: 'Analyze Disease',
          upload: 'Upload Image',
          capture: 'Capture Image',
          dragDrop: 'Drag & drop your soybean image here or click to browse',
          analyzing: 'Analyzing...',
          diseaseDetected: 'Disease Detected',
          confidence: 'Confidence',
          overview: 'Overview',
          actions: 'Actions',
          symptoms: 'Symptoms',
          treatment: 'Treatment',
          monitoring: 'Monitoring',
          fertilizers: 'Fertilizers',
          exportPDF: 'Export PDF',
          exportJSON: 'Export JSON',
          analyzeAnother: 'Analyze Another Image',
          home: 'Home',
          uploadNav: 'Upload',
          features: 'Features',
          results: 'Results',
          about: 'About',
          detectionMap: 'Detection mAP@0.5–0.95',
          ensembleAccuracy: 'Ensemble Accuracy',
          diseaseTypes: 'Disease Types',
          supportedClasses: 'Supported Classes',
          yoloDetector: 'YOLOv8 Detector',
          showAIFocus: 'Show AI Focus (Grad-CAM)',
          hideAIFocus: 'Hide AI Focus'
        }
      },
      hi: {
        name: 'हिन्दी',
        flag: '🇮🇳',
        ui: {
          title: 'सोयाबीन डिटेक्ट AI',
          analyze: 'रोग का विश्लेषण करें',
          upload: 'छवि अपलोड करें',
          capture: 'छवि कैप्चर करें',
          dragDrop: 'अपनी सोयाबीन की छवि यहां खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें',
          analyzing: 'विश्लेषण कर रहे हैं...',
          diseaseDetected: 'रोग का पता चला',
          confidence: 'विश्वास स्तर',
          overview: 'अवलोकन',
          actions: 'कार्रवाई',
          symptoms: 'लक्षण',
          treatment: 'उपचार',
          monitoring: 'निगरानी',
          fertilizers: 'उर्वरक',
          exportPDF: 'PDF निर्यात करें',
          exportJSON: 'JSON निर्यात करें',
          analyzeAnother: 'दूसरी छवि का विश्लेषण करें',
          home: 'होम',
          uploadNav: 'अपलोड',
          features: 'विशेषताएं',
          results: 'परिणाम',
          about: 'के बारे में',
          detectionMap: 'डिटेक्शन mAP@0.5–0.95',
          ensembleAccuracy: 'एन्सेम्बल सटीकता',
          diseaseTypes: 'रोग प्रकार',
          supportedClasses: 'समर्थित वर्ग',
          yoloDetector: 'YOLOv8 डिटेक्टर',
          showAIFocus: 'AI फोकस दिखाएं (Grad-CAM)',
          hideAIFocus: 'AI फोकस छिपाएं'
        }
      },
      mr: {
        name: 'मराठी',
        flag: '🇮🇳',
        ui: {
          title: 'सोयाबीन डिटेक्ट AI',
          analyze: 'रोगाचे विश्लेषण करा',
          upload: 'प्रतिमा अपलोड करा',
          capture: 'प्रतिमा कॅप्चर करा',
          dragDrop: 'तुमची सोयाबीन प्रतिमा येथे ड्रॅग आणि ड्रॉप करा किंवा ब्राउझ करण्यासाठी क्लिक करा',
          analyzing: 'विश्लेषण करत आहे...',
          diseaseDetected: 'रोग आढळला',
          confidence: 'विश्वास पातळी',
          overview: 'सारांश',
          actions: 'कृती',
          symptoms: 'लक्षणे',
          treatment: 'उपचार',
          monitoring: 'निरीक्षण',
          fertilizers: 'खते',
          exportPDF: 'PDF निर्यात करा',
          exportJSON: 'JSON निर्यात करा',
          analyzeAnother: 'दुसऱ्या प्रतिमेचे विश्लेषण करा',
          home: 'मुख्यपृष्ठ',
          uploadNav: 'अपलोड',
          features: 'वैशिष्ट्ये',
          results: 'परिणाम',
          about: 'आमच्याबद्दल',
          detectionMap: 'डिटेक्शन mAP@0.5–0.95',
          ensembleAccuracy: 'एन्सेम्बल अचूकता',
          diseaseTypes: 'रोग प्रकार',
          supportedClasses: 'समर्थित वर्ग',
          yoloDetector: 'YOLOv8 डिटेक्टर',
          showAIFocus: 'AI फोकस दाखवा (Grad-CAM)',
          hideAIFocus: 'AI फोकस लपवा'
        }
      }
    };
  }

  translate(key, lang = null) {
    const targetLang = lang || this.currentLang;
    const keys = key.split('.');
    let value = this.translations[targetLang];
    
    for (const k of keys) {
      value = value?.[k];
      if (!value) break;
    }
    
    return value || key;
  }

  switchLanguage(lang) {
    if (!this.translations[lang]) {
      console.warn(`Language ${lang} not supported`);
      return false;
    }
    
    this.currentLang = lang;
    localStorage.setItem('preferred_language', lang);
    this.updateUI();
    console.log(`Language switched to: ${this.translations[lang].name}`);
    return true;
  }

  updateUI() {
    // Update all UI elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      el.textContent = this.translate(key);
    });
    
    // Update placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      el.placeholder = this.translate(key);
    });
  }

  getAvailableLanguages() {
    return Object.keys(this.translations).map(code => ({
      code,
      name: this.translations[code].name,
      flag: this.translations[code].flag
    }));
  }

  translateDiseaseInfo(result, targetLang = null) {
    // Translate disease information (post-processing)
    const lang = targetLang || this.currentLang;
    
    if (lang === 'en') {
      return result; // No translation needed
    }
    
    // Disease name translations (12+ classes)
    const diseaseTranslations = {
      hi: {
        'Anthracnose': 'एन्थ्रेक्नोज',
        'Bacterial Blight': 'जीवाणु झुलसा',
        'Bacterial Pustule': 'जीवाणु पुस्टुल',
        'Brown Spot': 'भूरे धब्बे',
        'Cercospora Leaf Blight': 'सर्कोस्पोरा पत्ती झुलसा',
        'Downey Mildew': 'डाउनी मिल्ड्यू',
        'Frogeye Leaf Spot': 'फ्रॉगआई पत्ती धब्बा',
        'Healthy': 'स्वस्थ',
        'Insects': 'कीट',
        'Mosaic Virus': 'मोज़ेक वायरस',
        'Nutrient Deficiencies': 'पोषक तत्व की कमी',
        'Powdery Mildew': 'पाउडरी मिल्ड्यू',
        'Rust': 'रतुआ',
        'Southern Blight': 'दक्षिणी झुलसा',
        'Sudden Death Syndrome': 'अचानक मृत्यु सिंड्रोम',
        'Target Spot': 'लक्ष्य धब्बा',
        'Yellow Mosaic': 'पीला मोज़ेक'
      },
      mr: {
        'Anthracnose': 'अँथ्रॅक्नोज',
        'Bacterial Blight': 'जिवाणु झुलूस',
        'Bacterial Pustule': 'जिवाणु पुस्टुल',
        'Brown Spot': 'तपकिरी ठिपके',
        'Cercospora Leaf Blight': 'सर्कोस्पोरा पान झुलूस',
        'Downey Mildew': 'डाउनी मिल्ड्यू',
        'Frogeye Leaf Spot': 'फ्रॉगआय पान ठिपके',
        'Healthy': 'निरोगी',
        'Insects': 'कीड',
        'Mosaic Virus': 'मोझेक व्हायरस',
        'Nutrient Deficiencies': 'पोषक तत्वांची कमतरता',
        'Powdery Mildew': 'पावडरी मिल्ड्यू',
        'Rust': 'गंज',
        'Southern Blight': 'दक्षिणी झुलूस',
        'Sudden Death Syndrome': 'अचानक मृत्यू सिंड्रोम',
        'Target Spot': 'लक्ष्य ठिपके',
        'Yellow Mosaic': 'पिवळा मोझेक'
      }
    };
    
    // Create translated copy
    const translated = {...result};
    
    if (diseaseTranslations[lang] && diseaseTranslations[lang][result.CLASS]) {
      translated.CLASS_TRANSLATED = diseaseTranslations[lang][result.CLASS];
      translated.CLASS_ORIGINAL = result.CLASS;
    }
    
    // Note: Full text translation would require API or pre-translated database
    // For now, disease names are translated, full descriptions remain in English
    
    return translated;
  }
}

// Export for global use
window.I18NManager = I18NManager;
