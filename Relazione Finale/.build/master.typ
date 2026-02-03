
#import "master_template.typ": project

#show: project.with(
  title: "Relazione Finale",
  author: "Author Name",
  university: "",
  department: "",
  degree: "",
  supervisor: "",
  academic_year: "",
  degree_label: "Corso di laurea magistrale in",
  academic_year_label: "Anno Accademico",
  logo_path: none,
  numbered_chapters: true,
  heading_numbering: none,
  date: "2026-02-03",
  version: "0.1.0",
)


#pagebreak(weak: true)

= Analisi Architetturale e Teoria


#include "01_Analisi Architetturale e Teoria/01_Il problema dellAnomaly Detection Finanziaria.typ"

#include "01_Analisi Architetturale e Teoria/02_Autoencoder AE.typ"

#include "01_Analisi Architetturale e Teoria/03_Adversarial Autoencoder AAE.typ"

#pagebreak(weak: true)

= Il Progetto Originale Baseline


#include "02_Il Progetto Originale Baseline/01_Il Paper di Riferimento.typ"

#include "02_Il Progetto Originale Baseline/02_Il Dataset.typ"

#include "02_Il Progetto Originale Baseline/03_Pipeline Originale.typ"

#include "02_Il Progetto Originale Baseline/04_Riproduzione dei Risultati Baseline.typ"

#pagebreak(weak: true)

= Estensione e Metodologia


#include "03_Estensione e Metodologia/01_Modernizzazione dello Stack Tecnologico.typ"

#include "03_Estensione e Metodologia/02_Ottimizzazione Data Pipeline Memory Engineering.typ"

#include "03_Estensione e Metodologia/03_Design dello Stress Test Imbalance.typ"

#include "03_Estensione e Metodologia/04_Analisi di Stabilità Training Dynamics.typ"

#include "03_Estensione e Metodologia/05_Confronto Architetturale Ablation Study.typ"

#pagebreak(weak: true)

= Analisi dei Risultati e Discussione


#include "04_Analisi dei Risultati e Discussione/01_Performance di Classificazione MAUC.typ"

#include "04_Analisi dei Risultati e Discussione/02_Analisi Qualitativa t-SNE.typ"

#include "04_Analisi dei Risultati e Discussione/03_Stabilità del Training.typ"

#include "04_Analisi dei Risultati e Discussione/04_Robustezza allImbalance.typ"

#pagebreak(weak: true)

= Conclusioni


#include "05_Conclusioni/01_Conclusioni.typ"

#pagebreak(weak: true)

= Appendici


#include "06_Appendici/01_Appendice A Configurazione Ambiente.typ"

#include "06_Appendici/02_Appendice B Iperparametri.typ"

#pagebreak(weak: true)
#set heading(numbering: none)
#bibliography("references.bib")