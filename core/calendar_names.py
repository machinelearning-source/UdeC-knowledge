"""Official names of the 2026 academic calendar activities.

The app shows the English names; the calendar PDF uses the Spanish official
names, so retrieval looks activities up by the Spanish name.
"""

import unicodedata

POSTGRADUATE_NAME = "(Posgrados)"

CALENDAR_OFFICIAL_NAMES = {
    "Online registration": "Inscripciones En Línea",
    "Publication of shortlisted applicants": "Publicación Preseleccionados",
    "Admission test": "Prueba De Admisión",
    "Publication of admitted students lists": "Publicación De Listas Admitidos",
    "Institutional characterization - new regular students": "Caracterización Institucional Nuevos Regulares",
    "Institutional characterization - returning, transfer and re-entry students": "Caracterización Institucional Estudiantes Antiguos No Caracterizados - Transferencias - Reingresos",
    "Payment - new students": "Pago Estudiantes Nuevos",
    "Optional students": "Estudiantes Opcionales",
    "Ordinary tuition payment - returning students": "Pago Ordinario De Matricula Antiguos",
    "Publication of class schedules": "Publicación De Horarios",
    "Academic registration - new regular students": "Registro Académico Estudiantes Nuevos Regulares",
    "Academic registration - returning students": "Registro Académico Estudiantes Antiguos",
    "University life welcome experience": "Experiencia De Acogida A La Vida Universitaria",
    "Start of classes": "Inicio Clases",
    "Availability of student lists": "Disponibilidad De Listados De Estudiantes",
    "Adding of thematic units": "Adición De Núcleos Temáticos",
    "Cancellation of thematic units": "Cancelación De Núcleos Temáticos",
    "Semester cancellation": "Cancelación De Semestre",
    "Voluntary withdrawal": "Cancelación Retiro Voluntario",
    "CAI - Moment 1 (diagnostic)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 1 (Diagnóstico)",
    "CAI - Moment 2, remedial, first attempt (satisfactory)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Primera Vez (Satisfactorio)",
    "CAI - Moment 2, remedial, second attempt (satisfactory)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Segunda Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 1 (diagnostic)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 1 (Diagnóstico)",
    "CAI 21st Century Generation course - Moment 2, remedial, first attempt (satisfactory)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Primera Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 2, remedial, second attempt (satisfactory)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Segunda Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), first attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) -- Primera Vez",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), second attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Segunda Vez",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), third attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Tercera Vez",
    "Deadline for grade entry per term": "Fecha Limite Para Registro De Notas Por Corte",
    "University week": "Semana Universitaria",
    "End of academic period": "Finalización De Periodo Académico",
    "Closing process and setup of the new academic period": "Proceso Cierre Y Parametrización Nuevo Periodo Académico",
    "Registration for intersemester courses": "Inscripciones Cursos Intersemestrales",
    "Issuance of intersemester course receipts": "Expedición Recibos Cursos Intersemestrales",
    "Payment of intersemester course": "Pago Curso Intersemestral",
    "Start of intersemester courses": "Inicio Cursos Intersemestrales",
    "Grade entry for intersemester courses": "Registro De Notas Intersemestrales",
    "Document review period - ordinary degrees": "Periodo Revisión Documentos Para Grado Ordinarios",
    "Ordinary graduations": "Grados Ordinarios",
    "Document review period - extraordinary degrees": "Periodo Revisión Documentos Para Grado Extraordinarios",
    "Extraordinary graduations": "Grados Extraordinarios",
    "Application for transfers and re-entries": "Solicitud De Traslados, Transferencias Y Reingresos",
    "Deadline for approval of transfers and re-entries": "Fecha Limite De Aprobación Traslados, Transferencias Y Reingresos",
    "Deadline for reporting academic changes through SIS": "Fecha Máxima De Reporte De Novedades Académicas A Traves Del SIS (Traslados, Transferencias Y Reingresos)",
    "Deadline to register degree option / advanced semester": "Fecha Máxima Inscripción Opción De Trabajo De Grado/ Semestre Avanzado",
    "Deadline for approval of degree option / advanced semester": "Fecha Máxima De Aprobación Opción De Grado/ Semestre Avanzado",
    "Student evaluation of teachers": "Evaluación De Los Estudiantes A Los Docentes",
    "Teacher self-assessment": "Autoevaluación Docentes",
    "Coordinators' evaluation of teachers": "Evaluación De Los Coordinadores A Los Docentes",
    "Faculty vacations (tenured staff)": "Vacaciones Docentes Planta",
    "Student vacations": "Vacaciones Estudiantes",
    "Holy Week": "Semana Santa",
    "Online registration (postgraduate)": "Inscripciones En Línea (Posgrados)",
    "Selection interviews (postgraduate)": "Entrevistas De Selección (Posgrados)",
    "Publication of admitted students (postgraduate)": "Publicación Admitidos (Posgrados)",
    "Institutional characterization - new regular students (postgraduate)": "Caracterización Institucional Estudiantes Nuevo Regular - Posgrados (Posgrados)",
    "Tuition payment (postgraduate)": "Pago De Matricula (Posgrados)",
    "Homologation process - degree option / advanced semester (postgraduate)": "Porceso De Homologación (Opción De Titulación-Semestre Avanzado) (Posgrados)",
    "Academic registration (postgraduate)": "Registro De Matricula Académica (Posgrados)",
    "University life welcome experience (postgraduate)": "Experiencia De Acogida A La Vida Universtaria (Posgrados)",
    "Start of classes (postgraduate)": "Inicio Clases (Posgrados)",
    "End of academic period (postgraduate)": "Finalización De Periodo Académico (Posgrados)",
    "Closing process and setup of the new academic period (postgraduate)": "Proceso Cierre Y Parametrización Nuevo Periodo Académico (Posgrados)",
    "Document review period - ordinary degrees (postgraduate)": "Periodo Revisión Documentos Para Grados Ordinarios (Posgrados)",
    "Ordinary graduations (postgraduate)": "Grados Ordinarios (Posgrados)",
    "Document review period - extraordinary degrees (postgraduate)": "Periodo Revisión Documentos Para Grados Extraordinarios (Posgrados)",
    "Extraordinary graduations (postgraduate)": "Grados Extraordinarios (Posgrados)",
}


def _simplify(text):
    text = unicodedata.normalize("NFKD", text or "")
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).lower().split())


def calendar_name_in(question):
    """Return the official calendar name of the activity a question mentions, if any.

    Both the English and the Spanish names are recognised. A postgraduate
    activity is chosen only when the question mentions postgraduate studies.
    """
    simple_question = _simplify(question)
    asks_postgraduate = "postgrad" in simple_question or "posgrad" in simple_question
    candidates = []
    for english_name, official_name in CALENDAR_OFFICIAL_NAMES.items():
        if (POSTGRADUATE_NAME in official_name) != asks_postgraduate:
            continue
        spanish_name = official_name.replace(POSTGRADUATE_NAME, "").strip()
        english_name = english_name.replace("(postgraduate)", "").strip()
        for name in (english_name, spanish_name):
            if _simplify(name) in simple_question:
                candidates.append((len(name), official_name))
    return max(candidates)[1] if candidates else None
