# 🔬 CI Failure Annotation Tool

## 🚀 Quick Start

### **For Nakhla:**
```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/annotation
./start_nakhla.sh
```

### **For Rabeya:**
```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/annotation
./start_rabeya.sh
```

This will:
1. Start a local web server
2. Open the annotation tool in your browser
3. Load all 565 instances from your CSV

**To stop:** Press `Ctrl+C` in the terminal

---

## 📁 Files

```
annotation/
├── nakhla_annotate.html          Nakhla's UI
├── nakhla_annotations.csv         Nakhla's data
├── start_nakhla.sh               ⭐ Run this for Nakhla
│
├── rabeya_annotate.html          Rabeya's UI
├── rabeya_annotations.csv         Rabeya's data
├── start_rabeya.sh               ⭐ Run this for Rabeya
│
├── ANNOTATORS_GUIDE.md           Full guide
├── HOW_TO_ANNOTATE.txt           Taxonomy + examples
└── compare_annotations.py        Calculate kappa
```

---

## 💾 Auto-Save

After starting the tool:

1. Click **"Enable Auto-Save"** button
2. Choose where to save your CSV
3. Annotate instances
4. Each annotation auto-saves to your CSV!

Your progress is also saved in browser (can resume anytime)

---

## 🎯 For Each Instance You'll See:

**Left Panel:**
- Instance ID
- Repository
- Workflow name
- 🔗 GitHub diff link (clickable)
- **System Classification:**
  - Failure types (DeepSeek's classification)
  - Sub-types
  - Full evidence/description

**Right Panel (Your Annotations):**
- ☑️ Select failure types (checkboxes)
- ➕ Add custom types
- 🏷️ Click-to-add sub-types
- ✍️ Type custom sub-types
- 📝 Add notes

**Navigation:**
- Use "Next" / "Previous" buttons
- Or use keyboard: ← → arrow keys
- Progress bar shows completion

---

## 📊 After Both Complete

```bash
python3 compare_annotations.py \
  nakhla_annotations.csv \
  rabeya_annotations.csv
```

Shows Cohen's Kappa score and agreement analysis

---

## ⏱️ Estimated Time

~5-7 hours for 565 instances  
(Can pause and resume anytime!)

---

## 🆘 Troubleshooting

**UI shows empty data?**
- Make sure you ran the `start_*.sh` script
- Don't open HTML files directly (use the script)

**Port already in use?**
- Stop other servers: `lsof -ti:8001 | xargs kill`
- Or change port in start script

**Auto-save not working?**
- Use Chrome or Edge browser
- Progress still saved in browser
- Can export at end

---

## ✨ Features

- 🎨 Clean, intuitive interface
- ☑️ 12 standard failure types
- ➕ Add custom types if needed
- 🏷️ Click-to-add common sub-types
- ✍️ Type custom sub-types
- 💾 Auto-save to CSV
- 📊 Progress tracking
- ⌨️ Keyboard shortcuts
- 📖 Built-in taxonomy reference
- 🔄 Resume capability

---

**Happy Annotating! 🎯**
