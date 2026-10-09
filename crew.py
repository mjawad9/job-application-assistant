from crewai import Agent, Task, Crew, Process, LLM


def build_crew(cv_text: str, job_description: str, model: str) -> Crew:
    llm = LLM(model=model, temperature=0.3)

    # ---------- Agents ----------
    cv_analyzer = Agent(
        role="CV Analyzer",
        goal="Extract skills, experience, projects and weaknesses from a candidate's CV.",
        backstory="You are a senior technical recruiter who reads hundreds of CVs a week "
                  "and knows exactly what hiring managers look for.",
        llm=llm,
    )
    job_matcher = Agent(
        role="Job Matcher",
        goal="Compare the CV against the job description and score the fit honestly.",
        backstory="You are a hiring analyst. You are fair and specific, and you never "
                  "invent skills the candidate does not have.",
        llm=llm,
    )
    cover_letter_writer = Agent(
        role="Cover Letter Writer",
        goal="Write a tailored, genuine cover letter based only on facts from the CV.",
        backstory="You are a professional career writer who writes concise, "
                  "human-sounding letters without clichés.",
        llm=llm,
    )
    interview_coach = Agent(
        role="Interview Coach",
        goal="Prepare the candidate for likely interview questions for this role.",
        backstory="You are an experienced interviewer in AI and software engineering "
                  "who gives practical, honest coaching.",
        llm=llm,
    )

    # ---------- Tasks ----------
    analyze_task = Task(
        description=f"Analyze this CV and summarize the candidate's skills, experience, "
                    f"projects and any gaps or weak points.\n\nCV:\n{cv_text}",
        expected_output="A structured summary: skills, experience, projects, weaknesses.",
        agent=cv_analyzer,
    )
    match_task = Task(
        description=f"Using the CV analysis, compare the candidate to this job description.\n\n"
                    f"Job description:\n{job_description}\n\n"
                    "Give: a match score out of 100, matching skills, missing skills, "
                    "and 5 concrete CV improvements for this job.",
        expected_output="Match score, matching skills, missing skills, and CV improvements.",
        agent=job_matcher,
        context=[analyze_task],
    )
    letter_task = Task(
        description=f"Write a tailored cover letter (maximum 250 words) for this job.\n\n"
                    f"Job description:\n{job_description}\n\n"
                    "Use only real facts from the CV analysis. Do not invent experience.",
        expected_output="A ready-to-send cover letter.",
        agent=cover_letter_writer,
        context=[analyze_task, match_task],
    )
    interview_task = Task(
        description=f"Create 8 likely interview questions for this role (technical and "
                    f"behavioral).\n\nJob description:\n{job_description}\n\n"
                    "For each question, add a short tip on how this candidate should "
                    "answer, based on their CV.",
        expected_output="8 questions, each with an answer tip.",
        agent=interview_coach,
        context=[analyze_task, match_task],
    )

    return Crew(
        agents=[cv_analyzer, job_matcher, cover_letter_writer, interview_coach],
        tasks=[analyze_task, match_task, letter_task, interview_task],
        process=Process.sequential,
        verbose=False,
    )