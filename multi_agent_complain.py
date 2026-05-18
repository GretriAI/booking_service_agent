import os
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool

# Mocking API key environment variable setup
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'  # Recommended for complex multi-step reasoning

# Define custom tools to fetch Booking.com specific documentation or reservation systems
booking_policy_tool = ScrapeWebsiteTool(
    website_url="https://www.booking.com/content/terms.html"
)

# -------------------------------------------------------------------------
# Define Agents
# -------------------------------------------------------------------------

claims_agent = Agent(
    role="Senior Booking Claims Specialist",
    goal="Resolve customer claims regarding hotel overcharges, cancellations, and amenities discrepancies empathetically and accurately.",
    backstory=(
        "You work at Booking.com's Elite Escalations tier. You handle critical disputes "
        "between guests and hotel partners. Your objective is to cross-verify reservation logs, "
        "apply Booking.com compensation policies seamlessly, and provide absolute clarity without making assumptions."
    ),
    allow_delegation=False,
    verbose=True
)

claims_qa_agent = Agent(
    role="Claims Quality Assurance Auditor",
    goal="Ensure all claim resolutions comply strictly with legal compliance frameworks and internal financial refund structures.",
    backstory=(
        "You are an auditor overseeing claims settlements. You verify that the Senior Claims Specialist "
        "has concrete evidence of hotel fault before approving refunds or issuing Booking Wallet credits. "
        "You maintain a highly supportive but legally airtight tone."
    ),
    verbose=True
)

# -------------------------------------------------------------------------
# Define Tasks
# -------------------------------------------------------------------------

claim_investigation = Task(
    description=(
        "A guest ({customer_name}) has filed a claim regarding reservation ID {booking_id}.\n"
        "Claim Detail: {claim_details}\n"
        "Investigate the discrepancy against our standard policies. Draft a resolution outlining "
        "refund percentages, Booking credit options, or fallback verification steps if hotel confirmation is pending."
    ),
    expected_output=(
        "A highly structured dispute resolution breakdown containing an explanation of findings, "
        "financial compensation math (if any), concrete action items for the hotel, and next steps for the customer."
    ),
    tools=[booking_policy_tool],
    agent=claims_agent,
)

audit_and_approval = Task(
    description=(
        "Review the proposed claim settlement drafted by the Senior Claims Specialist for booking {booking_id}. "
        "Ensure the resolution logic aligns with Booking.com's liability caps, guarantees fair treatment "
        "to our hotel partners, and provides an exceptionally clear path forward without corporate jargon."
    ),
    expected_output=(
        "A polished, final claim resolution communication package ready to dispatch to the customer, "
        "explicitly specifying refund amounts, processing timelines, and confirmation references."
    ),
    agent=claims_qa_agent,
)

# -------------------------------------------------------------------------
# Assemble and Run Crew
# -------------------------------------------------------------------------

booking_claims_crew = Crew(
    agents=[claims_agent, claims_qa_agent],
    tasks=[claim_investigation, audit_and_approval],
    verbose=2,
    memory=True
)

# Example execution details
inputs = {
    "customer_name": "Sarah Jenkins",
    "booking_id": "BK-98471102",
    "claim_details": "The hotel charged me a $150 hidden resort fee at check-out that was explicitly stated as 'Included' on my Booking.com confirmation voucher. The hotel refused to remove it."
}

# result = booking_claims_crew.kickoff(inputs=inputs)