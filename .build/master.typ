
#import "master_template.typ": project

#show: project.with(
  title: "fraud_detect_AAE_effects-master",
  author: "Author",
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
  date: "2024",
  version: "1.0",
)


#pagebreak(weak: true)

= data


#pagebreak(weak: true)

= latent_data_sets


#pagebreak(weak: true)

= models


#pagebreak(weak: true)

= openspec


#include "openspec/AGENTS.typ"

#include "openspec/project.typ"

#pagebreak(weak: true)

= project_extension


#include "project_extension/recap.typ"

#include "project_extension/relazione-modifica ambiente.typ"

#include "project_extension/relazione_finale.typ"

#include "project_extension/relazione_tecnica_extension.typ"

#pagebreak(weak: true)

= Relazione Finale


#include "Relazione Finale/README.typ"

#pagebreak(weak: true)

= results


#pagebreak(weak: true)

= scripts


#pagebreak(weak: true)

= tsneResults


#pagebreak(weak: true)
#set heading(numbering: none)
#bibliography("references.bib")