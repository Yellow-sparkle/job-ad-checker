"""Job Ad Checker rules: severities, pre-written reasons and the verdict.

Severities (PRD "Scam rule severity", as changed in DECISIONS.md):
  sure    forces "Skip"
  strong  at least "Be careful" (GHOST_TALENT_POOL, STRONG_HIDE_FROM_AI and the
          agency DBS note also give at least "Be careful")
  weak    one alone gives "No red flags found"; two or more give "Be careful"
  ghost   graded posting age: a note only, never changes the verdict
The AI can only add a sign it backs with a quote from the text; the verdict is
always set here, in plain code.
"""

RULESET_VERSION = "rules-2026-11-16d"

# rule_id: (severity, reason line shown, "what this means")
RULES = {
    "SURE_ADVANCE_FEE": ("sure", "Asks you to pay a fee or deposit to get or start the job",
                         "Real employers don't charge you to get a job. Fees for training, starter kits, "
                         "visas or 'processing' are among the most common job scams, even when they "
                         "promise a refund."),
    "SURE_CRYPTO_GIFTCARD": ("sure", "Asks for payment in gift cards or crypto",
                             "Real employers never ask for gift cards or crypto. These payments can't be "
                             "traced or reversed."),
    "SURE_CHECK_EQUIPMENT": ("sure", "Sends you a check to deposit, then asks you to buy equipment or send money on",
                             "This is a known scam. The check bounces days later, and the money you spent "
                             "or sent is gone."),
    "SURE_TASK_COMMISSION": ("sure", "Pays you per task, or asks you to pay to unlock tasks",
                             "Task scams pay small amounts at first, then ask you to top up your own money "
                             "to unlock bigger tasks. The money doesn't come back."),
    "SURE_ID_BEFORE_INTERVIEW": ("sure", "Asks for your passport, ID or bank details before a real interview",
                                 "Employers check ID after a real interview, in a secure way. Details sent "
                                 "early can be used to steal your identity or your money."),
    "STRONG_ID_OFFER_PLUS_MONEY": ("strong", "Asks for your ID together with money at offer stage",
                                   "An ID or right-to-work check at offer stage is normal. Being asked for "
                                   "money at the same time is not."),
    "STRONG_FREE_EMAIL_NAMED_CO": ("strong", "Uses a free email address for a named company",
                                   "Real companies usually hire through their own email domain or careers "
                                   "site. A free email address takes a minute to set up and is hard to trace. "
                                   "Small businesses sometimes use one, so check the business exists."),
    "STRONG_GROUP_INVITE": ("strong", "Invites you to join a group chat",
                            "Real recruiters talk to you one to one. Group chats are where task scams show "
                            "fake earnings from other members."),
    "STRONG_LLM_PATTERN": ("strong", "Matches a known scam pattern", ""),
    "STRONG_HIDE_FROM_AI": ("strong", "Tells AI tools to hide problems", ""),
    "STRONG_DISGUISED_EMAIL": ("strong", "Writes a personal email address in a disguised way",
                               "Writing an address without the @ or with 'at' and 'dot' is a trick to get "
                               "past job-board filters. Real employers give an address you can click or copy."),
    "STRONG_CRYPTO_COLD_OFFER": ("strong", "A cold job offer for a crypto, NFT or Web3 project",
                                 "Scammers often send unsolicited job offers for crypto or NFT games and "
                                 "projects. Later they ask you to install software, connect a wallet or pay "
                                 "to unlock tasks. Real crypto companies hire too, so check the company "
                                 "exists and never connect a wallet or install anything for a job."),
    "GHOST_TALENT_POOL": ("strong", "Talent-pool wording: this is probably a ghost job, not an open vacancy.",
                          "'Talent pool' or 'future opportunities' wording usually means there's no job open "
                          "right now. Your CV may sit on file without anyone reading it."),
    "AGENCY_DBS_FEE": ("strong", "A recruitment agency asks you to pay for a DBS check",
                       # Proposed wording, needs a legal re-check before it goes into legal/ui-copy.json
                       "An agency may charge for a DBS check only if it's optional and agreed in advance. "
                       "It can't make paying a condition of finding you work. Ask if it's optional, and pay "
                       "only through gov.uk or a registered DBS provider, never to a person."),
    # Kate, 14-15 Nov (GS-R10): these three together give Skip, though each alone is not sure
    "SURE_FREE_EMAIL_NO_INTERVIEW_SAME_DAY": ("sure",
        "A free email for a named company, no interview and a same-day deadline, all together",
        "Each of these is a warning sign on its own. Together they match a common fake-job pattern: "
        "a big name, no way to check who you're talking to, no interview and pressure to act today."),
    "WEAK_CHAT_APP": ("weak", "Moves you to a chat app",
                      "Moving to a chat app takes you off the platform that could protect you and makes the "
                      "sender hard to trace. Some real agencies do this too, so on its own it's a weak sign."),
    "WEAK_NO_INTERVIEW": ("weak", "No interview or call is mentioned",
                          "Real employers almost always talk to you before they hire you. Jobs that skip "
                          "this step are common in scams."),
    # Kate, 16 Nov (real ad RA11, goo.gl link): found in plain code
    "WEAK_SHORT_LINK": ("weak", "Uses a shortened link",
                        "A shortened link (like bit.ly or goo.gl) hides where it really goes. Real employers "
                        "usually link to their own website or a job board."),
    # Kate, 16 Nov (real ad RA20): a "big pay for little online work" finding whose quote names no tasks
    # Doesn't rate the pay itself (12 Nov: the checker never calls pay high or low).
    "WEAK_PAY_NOT_FIT": ("weak", "Looks a little like a paid-online-work scam",
                         "The AI saw a partial match with paid-online-work scams, but the text names no tasks, "
                         "so on its own it's a weak sign. Check the job with the employer directly."),
    # Kate, 16 Nov (real ad RA20): found in plain code, so it doesn't depend on the AI
    "WEAK_VOLUNTEER_WITH_PAY": ("weak", "Calls the role volunteer work but also states pay",
                                "Volunteer roles are unpaid. An ad that says both is either careless or "
                                "using a big number to get you to reply. Check the role with the organisation."),
    "WEAK_URGENCY": ("weak", "Pushes you to act fast",
                     "Pressure to reply, pay or start right away leaves you less time to check the job."),
    # Kate, 14 Nov: "the text you pasted", so it stays true on partial pastes (GS-R12)
    "WEAK_NO_EMPLOYER_NAMED": ("weak", "The text you pasted doesn't say who the employer is.",
                               "Without a name, you can't check the job on the employer's own website."),
}

PATTERN_LABELS = {  # fixed, versioned list for the AI pattern check
    "reshipping": "receiving and re-sending parcels (reshipping)",
    "money_mule": "moving money through your own account (money mule)",
    "task_scam": "paid online tasks (task scam)",
    "pay_to_recruit": "pay that comes from recruiting other people",
    "data_harvest": "asking for personal details early",
    "fake_research_assistant": "a 'professor' offering paid remote research work by text (fake research assistant)",
}
PATTERN_LIST_VERSION = "patterns-2026-11-16"
# Patterns that give Skip on their own (Kate, 16 Nov). The rest stay "at least Be careful".
SKIP_PATTERNS = {"money_mule", "task_scam", "fake_research_assistant"}

# Fixed note for the personal-details pattern; the AI's own note isn't shown (debate, 14 Nov)
DATA_HARVEST_NOTE = "Asks for personal details early; check why they need them before you send anything."

GHOST_AGE_30 = "Posted 30+ days ago. Ghost jobs are more common at this age."
GHOST_AGE_60 = "Posted 60+ days ago. This is more likely to be a ghost job."

DBS_EMPLOYER_NOTE = ("Some UK employers ask new starters to pay for their DBS check. Pay only through gov.uk "
                     "or the employer's registered DBS provider, never to a person.")

VERDICT_SKIP, VERDICT_CAREFUL, VERDICT_OK = "Skip", "Be careful", "No red flags found"
VERDICT_ASK_TEXT = "No verdict (asks for text)"
VERDICT_NOT_ENGLISH = "Out of scope (English only)"

WHAT_WE_CHECKED = ("We checked for upfront fees, gift-card or crypto payments, fake checks, paid-task schemes, "
                   "early requests for ID or bank details, free email addresses for named companies, group "
                   "chats and chat apps, shortened links, hiring without an interview, pressure to hurry, AI-aimed lines, and "
                   "ghost-job signs. We didn't check the company itself.")


def severity(rule_id):
    return RULES[rule_id][0]


def verdict_for(fired, skip_pattern=False):
    """fired: list of rule ids. skip_pattern: the AI matched one of SKIP_PATTERNS (Kate, 16 Nov),
    which gives Skip although STRONG_LLM_PATTERN is otherwise a strong sign. Returns the verdict."""
    sev = [severity(r) for r in fired]
    if "sure" in sev or skip_pattern:
        return VERDICT_SKIP
    if "strong" in sev or sev.count("weak") >= 2:
        return VERDICT_CAREFUL
    return VERDICT_OK
