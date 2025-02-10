import argparse
import asyncio
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

from globalog import LOG
# Example of a LangGraph-based CrossClimb puzzle with loops/conditional edges.
# The puzzle consists of rung words plus top/bottom words, each with feedback.
# If rung guesses are incorrect, we re-generate rung clues. Similarly, for top/bottom.
# Finally, we evaluate puzzle difficulty.

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm

from sulamilim.dataprep.scrapping.fetch import fetch_content
from sulamilim.dataprep.scrapping.llm_reader import parse_for_llm


########################################
# PROMPTS
########################################

_SINGLE_WORD_CLUE_EXAMPLES_STR = """Example for clues are:
CORN - Vegtable that might be eaten "on the cob"
WORE - ____ one's heart on one's sleeve (was open about one's emotions)
FORE - Golfer's warning (and, a homophone of a single-digit number)
FORK - A utensil not suitable for eating soup
"""


_SINGLE_HEBREW_EXAMPLES_STR = """
- אבדה: דבר מה שנעלם
- אבוב: כמו גלגל מים גדול
- דאון: כלי טיס קל, בדר״כ ללא מנוע
- אתון: מאכילה את העיר בינקותו
"""


_TOP_BOTTOM_WORDS_CLUE_EXAMPLES = """
- CORD, WIRE: The top + bottom rows = Two terms for "a cable that can carry electricity."
- CLAY, SOIL: The top + bottom rows = A term for dirt, and a possible component of dirt.
"""


_TOP_BOTTOM_HEBREW_EXAMPLES = """
- גלימה, חליפה: נלבשים על הגוף - האחד לבוש רשמי והשני מלכותי ועשוי גם לשמש גיבורי על
- הונאה, התראה: מילה אחת מתארת מעשה זיוף, והשניה הודעת אזהרה שמקבלים (מהבנק למשל) אם עלו על כך.
- אפונה, שעורה: שני גידולי שדה, האד גדל בתרמילים והשני משמש לבישול בירה
- צליפה, קליעה: שני מונחים נרדפים המתארים פגיעה בול במטרה
"""


_SYSTEM_PROMPT_WORD_CLUES = (
    "אתה עוזר חכם ליצירת רמזים לחידת מילים בעברית. "
    "שים לב שהמילים הן בעברית ולכן חשוב שהרמזים יהיו גם בעברית, "
    "תוך שימוש במידע הרלוונטי (הגדרות, משמעויות) המתואר בקטע הבא. "
    "צור רמז קצר ומדוייק שמתאר או רומז למילה. הימנע מלהפוך את הרמז לפתרון גלוי. "
    "נסה להיות יצירתי, במקרים של דו משמעות תוכל להתייחס לאספקט אחד של המילה או כמה, תוכל לתת ביטוי עם חלק חסר "
    "כאשר המילמה משלימה את החלק החסר בביטוי. אם מופיע משוב מהרמז הקודם, התחשב בו ותקן."
    "\n\nדוגמאות לרמזים:"
    f"{_SINGLE_HEBREW_EXAMPLES_STR}"
    "\n\nועד דוגמאות מאנגלית בשביל רעיונות"
    f"{_SINGLE_WORD_CLUE_EXAMPLES_STR}"
)


_SYSTEM_PROMPT_TOP_BOTTOM_CLUES = (
    "אתה עוזר חכם ליצירת רמזים לחידת מילים בעברית. "
    "חבר רמז משותף לשתי מילים בעברית. "
    "כתוב רמז קצר בעברית הקושר בין שתי המילים."
    "שים לב שהמילים הן בעברית ולכן חשוב שהרמזים יהיו גם בעברית, "
    "תוך שימוש במידע הרלוונטי (הגדרות, משמעויות) המתואר בקטע הבא. "
    "צור רמז קצר ומדוויק, יכול להיות עם שני חלקים שמתאר או רומז לזוג המילים. הימנע מלהפוך את הרמז לפתרון גלוי. "
    "נסה להיות יצירתי, במקרים של דו משמעות תוכל להתייחס לאספקט אחד של המילה או כמה, תוכל לתת ביטוי עם חלק חסר "
    "כאשר המילה משלימה את החלק החסר בביטוי. אם מופיע משוב מהרמז הקודם, התחשב בו ותקן."
    "\n\nדוגמאות לרמזים:"
    f"{_TOP_BOTTOM_HEBREW_EXAMPLES}"
    "\n\nועד דוגמאות מאנגלית בשביל רעיונות"
    f"{_TOP_BOTTOM_WORDS_CLUE_EXAMPLES}"
)


########################################
# DATA MODELS
########################################

@dataclass
class BadFeedback:
    clue: str
    incorrect_answer: str

@dataclass
class Rung:
    word: str
    definition_text: str = ''
    clue: str = ''
    solved_answer: str = ''
    answer_explanation: str = ''
    # If solver guessed incorrectly, store feedback items
    feedbacks: List[Optional[BadFeedback]] = field(default_factory=list)

@dataclass
class TopBottomRung:
    top_word: str
    bottom_word: str
    top_definition_text: str = ''
    bottom_definition_text: str = ''
    clue: str = ''
    solved_top: str = ''
    solved_bottom: str = ''
    feedbacks: List[BadFeedback] = field(default_factory=list)

@dataclass
class PuzzleState:
    rung_words: List[Rung] = field(default_factory=list)
    top_bottom_rungs: Optional[TopBottomRung] = None

    # Whether rung words / top-bottom words are solved this round
    words_solved: bool = False
    top_bottom_solved: bool = False

    puzzle_difficulty: str = ''  # final rating
    structured_output: dict[str, Any] = field(default_factory=dict)


########################################
# LLMs
########################################

llm_clue_gen = ChatOpenAI(model='o1')   #, temperature=0.7)
llm_solver = ChatOpenAI(model='o1') #, temperature=0.5)
llm_difficulty = ChatOpenAI(model='o1') #, temperature=0.3)


########################################
# GENERATE CLUES
########################################

def generate_rung_clues(state: PuzzleState) -> PuzzleState:
    """
    Generate or re-generate rung clues. If feedback exists for a rung, incorporate it.
    """
    system_msg = SystemMessage(content=_SYSTEM_PROMPT_WORD_CLUES)
    for rung in state.rung_words:
        # Build a feedback summary:
        if rung.feedbacks:
            if rung.feedbacks[-1] is None:
                continue

            # Use the most recent feedback item
            previous_incorrect = '\n'.join(
                f" - {fb.clue} => {fb.incorrect_answer}"
                for fb in rung.feedbacks if fb is not None
            )
            user_prompt = (
                f"מילה: {rung.word}\n"
                f"רמזים קודמים שהובילו לתושובת שגויות :"
                f"\n{previous_incorrect}"
                "לפי תוכן (Markdown):\n" + rung.definition_text + "\n"
                "צור רמז חדש בעברית, נסה לדייק יותר אבל לא לגלות מיד."
            )
        else:
            # No feedback yet
            user_prompt = (
                f"מילה: {rung.word}\n"
                "תוכן (Markdown):\n" + rung.definition_text + "\n"
                "צור רמז עברי קצר שמתאים למילה זו."
            )

        LOG.info(f"[Generate-word-clues] generate clue for word \"{rung.word}\" with prompt of size {len(user_prompt)}")
        resp = llm_clue_gen.invoke([
            system_msg,
            HumanMessage(content=user_prompt)
        ])
        rung.clue = resp.content.strip()
        LOG.info(f"[Generate-word-clues] Got response: {rung.clue}")

    return state


def generate_top_bottom_clue(state: PuzzleState) -> PuzzleState:
    """
    Generate or re-generate a single clue for the top/bottom words.
    If there's any feedback, incorporate it.
    """
    tb = state.top_bottom_rungs
    system_msg = SystemMessage(content=_SYSTEM_PROMPT_WORD_CLUES)

    if tb and tb.feedbacks:
        latest_fb = tb.feedbacks[-1]
        user_prompt = (
            f"מילה עליונה: {tb.top_word}\n"
            f"מילה תחתונה: {tb.bottom_word}\n"
            f"הרמז הקודם היה: {latest_fb.clue}, "
            f"והתשובה השגויה: {latest_fb.incorrect_answer}\n\n"
            "מבוסס על תוכן המילה העליונה:\n" + tb.top_definition_text + "\n\n"
            "ותוכן המילה התחתונה:\n" + tb.bottom_definition_text + "\n"
            "צור רמז חדש משופר אך לא מספיילר."
        )
    else:
        user_prompt = (
            f"מילה עליונה: {tb.top_word}\n"
            f"מילה תחתונה: {tb.bottom_word}\n"
            "צור רמז משותף קצר בעברית, בלי לגלות ישירות.\n"
            "תוכן עליונה:\n" + tb.top_definition_text + "\n"
            "תוכן תחתונה:\n" + tb.bottom_definition_text + "\n"
        )

    tb = state.top_bottom_rungs
    LOG.info(f"[Generate-top-bottom-clues] generate clues for words \"{tb.top_word}, {tb.bottom_word}\" with prompt of size {len(user_prompt)}")
    resp = llm_clue_gen.invoke([
        system_msg,
        HumanMessage(content=user_prompt)
    ])
    tb.clue = resp.content.strip()
    LOG.info(f"[Generate-top-bottom-clues] Got response: {tb.clue}")
    return state


########################################
# SOLVE STEPS
########################################

def construct_words_solving_prompt(state: PuzzleState) -> tuple[str, list[int]]:
    puzzle_instructions = (
        "אתה מקבל רשימת רמזים בעברית עבור סולם מילים (word ladder). "
        "המשימה שלך היא לזהות את המילים המתאימות לכל רמז, ולסדר את המילים לפי סדר, בהתחשב בכך "
        "שכל מילה שונה רק באות אחת מהמילה הבאה. כמובן שגם זה רמז למילים אפשריות היכולות לשמש כתשובה\n\n"
        "ֿ\nהרמזים:\n"
    )

    clues_order = list(range(len(state.rung_words)))
    random.shuffle(clues_order)
    for i, rung_index in enumerate(clues_order):
        puzzle_instructions += f"{i + 1}. רמז: {state.rung_words[rung_index].clue}\n"

    puzzle_instructions += (
        "\n\nספק את הפתרונות למילים, וסדר אותן לפי סדר. תן את הערכתך הטובה ביותר."
        "בתשובתך שים כל מילה בשורה נפרדת, וסדר את המילים לפי סדר (עולה או יורד הכיוון לא חשוב)  "
        "העיקר שכל מילה תיבדל באות אחת מהמילים לצידה, "
        "ולצד כל מילה רשום את מספר המילה המקורי (לפני הסידור) כפי שנכתב בהוראות האלה."
        "בכל שורה כתוב: '{index}. {{מילה}}: {{הסבר}}'\n"
    )

    return puzzle_instructions, clues_order


def resolve_words_solving_response(response_text: str, clues_order: list[int], state: PuzzleState) -> PuzzleState:
    lines = response_text.splitlines()
    for line in lines:
        if '.' in line:
            splitted = line.split('.', maxsplit=1)
            clue_index_str = splitted[0].strip()
            if not clue_index_str.isdigit():
                continue

            clue_index = int(clue_index_str)
            after_dot = splitted[1].strip()
            if ':' in after_dot:
                guess_part, explanation = after_dot.split(':', 1)
            else:
                guess_part, explanation = after_dot, ''

            guess_word = guess_part.strip()
            explanation = explanation.strip()

            if 1 <= clue_index <= len(clues_order):
                rung_index = clues_order[clue_index - 1]
                rung = state.rung_words[rung_index]
                rung.solved_answer = guess_word
                rung.answer_explanation = explanation

    return state


def construct_top_bottom_solving_prompt(state: PuzzleState) -> str:
    tb = state.top_bottom_rungs
    solver_instructions = ("לפניך חידה בה צריך לגלות שתי מילים שונות. מילה ראשונה ומילה שניה."
                           "על מנת לגלות את שתי המילים מובאים שני סוגי אינפורמציה:"
                           "\n- רמז משותף לשתי המילים שעוזר לפתור את שתיהן."
                           "\n- לצד כל מילה יש מילה נוספת (לא בהכרח קשורה) הנבדלת ממנה באות אחת בלבד."
                           "\n\nלהלן הרמז המשותף:"
                           f"  {tb.clue}"
                           f"\n\nלהלן המילה הנוספת לצד כל אחת מהתשובות - מילת התשובה צריכה להבידל באות אחת מהמילה שלצידה:"
                           f"\n1. {state.rung_words[0].word}"
                           f"\n2. {state.rung_words[-1].word}"
                           "\n\n ציין בשתי שורות: '1. {{מילה עליונה}}', '2. {{מילה תחתונה}}' "
                           )
    return solver_instructions


def resolve_top_bottom_response(response_text: str, state: PuzzleState) -> PuzzleState:
    tb = state.top_bottom_rungs
    lines = response_text.splitlines()
    for line in lines:
        if line.startswith('1.'):
            guess = line[2:].strip(':').strip()
            tb.solved_top = ''.join(filter(str.isalpha, guess))
        elif line.startswith('2.'):
            guess = line[2:].strip(':').strip()
            tb.solved_bottom = ''.join(filter(str.isalpha, guess))

    return state


def solve_rungs(state: PuzzleState) -> PuzzleState:
    """
    Ask the LLM to solve rung clues.
    """
    puzzle_instructions, clues_order = construct_words_solving_prompt(state)
    LOG.info(f"[Solve-words] Sent solve request with prompt of size {len(puzzle_instructions)}")
    resp = llm_solver.invoke([HumanMessage(content=puzzle_instructions)])
    LOG.info(f"[Solve-words] Got response: {resp.content.strip()}")
    return resolve_words_solving_response(resp.content.strip(), clues_order, state)


def solve_top_bottom(state: PuzzleState) -> PuzzleState:
    solver_instructions = construct_top_bottom_solving_prompt(state)
    LOG.info(f"[Solve-top-bottom] Sent solve request with prompt of size {len(solver_instructions)}")
    resp = llm_solver.invoke([HumanMessage(content=solver_instructions)])
    LOG.info(f"[Solve-top-bottom] Got response: {resp.content.strip()}")
    state = resolve_top_bottom_response(resp.content.strip(), state)
    return state


########################################
# VERIFICATION STEPS
########################################

def verify_rungs(state: PuzzleState) -> PuzzleState:
    # Reset the flag
    LOG.info("Perform words verification)")
    state.words_solved = True
    for rung in state.rung_words:
        if rung.solved_answer.strip() != rung.word.strip():
            # store feedback
            LOG.info(f"Incorrect answer: {rung.word=} ; {rung.solved_answer=}")
            rung.feedbacks.append(
                BadFeedback(clue=rung.clue, incorrect_answer=rung.solved_answer)
            )
            state.words_solved = False
        else:
            rung.feedbacks.append(None)
    return state


def verify_top_bottom(state: PuzzleState) -> PuzzleState:
    # Reset the flag
    LOG.info("Perform top-bottom verification)")
    state.top_bottom_solved = True
    tb = state.top_bottom_rungs
    if tb.solved_top.strip() != tb.top_word.strip():
        LOG.info(f"Incorrect answer: {tb.top_word=} ; {tb.solved_top=}")
        tb.feedbacks.append(
            BadFeedback(clue=tb.clue, incorrect_answer=tb.solved_top)
        )
        state.top_bottom_solved = False

    if tb.solved_bottom.strip() != tb.bottom_word.strip():
        LOG.info(f"Incorrect answer: {tb.bottom_word=} ; {tb.solved_bottom=}")
        tb.feedbacks.append(
            BadFeedback(clue=tb.clue, incorrect_answer=tb.solved_bottom)
        )
        state.top_bottom_solved = False

    return state


########################################
# DIFFICULTY
########################################

def evaluate_puzzle_difficulty(state: PuzzleState) -> PuzzleState:
    puzzle_desc = "רשימת מילים:\n"
    for rung in state.rung_words:
        puzzle_desc += f"- מילה: {rung.word}, רמז: {rung.clue}\n"
    puzzle_desc += (f"\nמילה עליונה: {state.top_bottom_rungs.top_word}, "
                    f"מילה תחתונה: {state.top_bottom_rungs.bottom_word}\n"
                    f"רמז משותף: {state.top_bottom_rungs.clue}\n")

    system_msg = SystemMessage(content=(
        "אתה מעריך כעת את רמת הקושי הכללית של החידה. "
        "דרג מ-1 עד 5 (מספר בלבד) ואז שורה נוספת 'הסבר: ...'"
    ))
    user_msg = HumanMessage(content=puzzle_desc)

    LOG.info(f"[Puzzle-difficulty] Sent puzzle difficulty request with prompt of size {len(puzzle_desc)}")
    resp = llm_difficulty.invoke([system_msg, user_msg])
    LOG.info(f"[Puzzle-difficulty] Got response: {resp.content.strip()}")
    state.puzzle_difficulty = resp.content.strip()
    return state

########################################
# ROUTING
########################################

def rung_solved_router(state: PuzzleState) -> str:
    # If rung words not solved, go to 'generate_rung_clues', else proceed to top/bottom
    if state.words_solved:
        next_node = 'generate_top_bottom_clue'
    else:
        next_node = 'generate_rung_clues'

    LOG.info(f"[Routing - Words solved or not?] -> {next_node=}")
    return next_node


def top_bottom_solved_router(state: PuzzleState) -> str:
    # If top/bottom not solved, re-generate clue, else evaluate puzzle difficulty
    if state.top_bottom_solved:
        next_node = 'evaluate_puzzle_difficulty'
    else:
        next_node = 'generate_top_bottom_clue'

    LOG.info(f"[Routing - Top-Bottom solved or not?] -> {next_node=}")
    return next_node


########################################
# FINALIZING
########################################
def prepare_structured_output(state: PuzzleState) -> PuzzleState:
    """
    Collate everything into a dictionary for easy reading or JSON output.
    """
    for i, rung in enumerate(state.rung_words, 1):
        state.structured_output[f'word_{i}'] = {
            "word:": rung.word,
            "clue": rung.clue,
            "solved_answer": rung.solved_answer,
            "explanation": rung.answer_explanation,
        }

    state.structured_output['top_bottom'] = {
        'top_word': state.top_bottom_rungs.top_word,
        'bottom_word': state.top_bottom_rungs.bottom_word,
        'clue': state.top_bottom_rungs.clue,
        'solved_top': state.top_bottom_rungs.solved_top,
        'solved_bottom': state.top_bottom_rungs.solved_bottom,
    }

    state.structured_output['difficulty'] = state.puzzle_difficulty
    return state

########################################
# INITIALIZATION UTILITIES
########################################
def fetch_definition_as_markdown(url: str) -> str:
    """
    For demonstration, fetch the URL content as text,
    assume the entire content can be directly used in the LLM.
    Real use-case might parse the HTML or handle errors, etc.
    """
    absolute_url = "https://he.wiktionary.org/" + url
    try:
        content = fetch_content(absolute_url)
        md = parse_for_llm(content)
        return md
    except Exception as e:
        LOG.error(f"An error occurred during url parsing: {url}", exc_info=e)
        return "תוכן הדף לא נמצא"


def load_words_urls(path: Path) -> dict[str, str]:
    word_urls = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['word', 'url'])
        for row in tqdm(reader):
            w = row['word'].strip()
            u = row['url'].strip()
            word_urls[w] = u

    return word_urls


def fetch_word_info(word: str, words_urls: dict[str, str]) -> str:
    url = words_urls.get(word)
    if url is None:
        return "לא נמצאה כתובת."

    return fetch_definition_as_markdown(url)


def construct_initial_puzzle_state(words: str, top: str, bottom: str, word_urls_path: Path) -> PuzzleState:
    LOG.info(f"Loading words URLs from {word_urls_path}")
    word_urls = load_words_urls(word_urls_path)
    LOG.info(f"Loaded {len(word_urls)} words and URLs")


    rung_list = []
    # add rung words
    words = [w.strip() for w in words.split(',')]
    for w in words:
        # fetch definition
        definition_md = fetch_word_info(w, word_urls)
        rung_list.append(Rung(word=w, definition_text=definition_md))

    top_info = fetch_word_info(top, word_urls)
    bottom_info = fetch_word_info(bottom, word_urls)
    top_bottom_rungs = TopBottomRung(
        top_word=top,
        bottom_word=bottom,
        top_definition_text=top_info,
        bottom_definition_text=bottom_info,
    )

    puzzle_state = PuzzleState(
        rung_words=rung_list,
        top_bottom_rungs=top_bottom_rungs
    )

    return puzzle_state


########################################
# MAIN WORKFLOW
########################################
async def main():
    parser = argparse.ArgumentParser(description="Generate Hebrew CrossClimb clues via agentic flow.")
    parser.add_argument('--words', '-w', required=True,
                        help='Sequence of Hebrew words for the word ladder separated by a coma (excluding top/bottom).')
    parser.add_argument('--top', '-t', required=True, help='Top word in Hebrew.')
    parser.add_argument('--bottom', '-b', required=True, help='Bottom word in Hebrew.')
    parser.add_argument('--csv_path', '-f', type=Path, required=True,
                        help='Path to a CSV file with columns: [word,url] for definitions.')
    args = parser.parse_args()

    puzzle_state = construct_initial_puzzle_state(args.words, args.top, args.bottom, args.csv_path)

    workflow = StateGraph(PuzzleState)

    # Add nodes
    workflow.add_node('generate_rung_clues', generate_rung_clues)
    workflow.add_node('solve_rungs', solve_rungs)
    workflow.add_node('verify_rungs', verify_rungs)
    # workflow.add_node('rung_solved_router', rung_solved_router)

    workflow.add_node('generate_top_bottom_clue', generate_top_bottom_clue)
    workflow.add_node('solve_top_bottom', solve_top_bottom)
    workflow.add_node('verify_top_bottom', verify_top_bottom)
    # workflow.add_node('top_bottom_solved_router', top_bottom_solved_router)

    workflow.add_node('evaluate_puzzle_difficulty', evaluate_puzzle_difficulty)
    workflow.add_node('prepare_structured_output', prepare_structured_output)

    # Edges for rung flow
    workflow.add_edge(START, 'generate_rung_clues')
    workflow.add_edge('generate_rung_clues', 'solve_rungs')
    workflow.add_edge('solve_rungs', 'verify_rungs')

    # after verifying rung correctness, we route conditionally
    workflow.add_conditional_edges(
        source='verify_rungs',
        path=rung_solved_router,
        path_map={
            'generate_rung_clues': 'generate_rung_clues',
            'generate_top_bottom_clue': 'generate_top_bottom_clue'
        }
    )

    # Edges for top/bottom flow
    workflow.add_edge('generate_top_bottom_clue', 'solve_top_bottom')
    workflow.add_edge('solve_top_bottom', 'verify_top_bottom')
    # after verifying top/bottom, route conditionally
    workflow.add_conditional_edges(
        source='verify_top_bottom',
        path=top_bottom_solved_router,
        path_map={
            'generate_top_bottom_clue': 'generate_top_bottom_clue',
            'evaluate_puzzle_difficulty': 'evaluate_puzzle_difficulty'
        }
    )

    # finally, from evaluate_puzzle_difficulty -> END
    workflow.add_edge('evaluate_puzzle_difficulty', 'prepare_structured_output')
    workflow.add_edge('prepare_structured_output', END)

    workflow.set_entry_point('generate_rung_clues')
    workflow.set_finish_point('prepare_structured_output')

    app = workflow.compile()

    print(app.get_graph().draw_mermaid())

    final_state = app.invoke(puzzle_state)

    print(final_state['structured_output'])
    out = json.dumps(final_state['structured_output'], indent=2)
    print(out)

    print("\n===== Final Puzzle Results =====\n")

    for rung in final_state['rung_words']:
        print(f"Word: {rung.word} | Clue: {rung.clue}")
        print(f"Solved as: {rung.solved_answer}")
        if rung.feedbacks:
            print(f"Feedbacks: {[ (fb.clue, fb.incorrect_answer) for fb in rung.feedbacks ]}")
        print("---")

    tb = final_state.top_bottom_rungs
    print(f"Top: {tb.top_word}, solved as {tb.solved_top}")
    print(f"Bottom: {tb.bottom_word}, solved as {tb.solved_bottom}")
    if tb.feedbacks:
        print(f"Top/Bottom Feedbacks: {[ (fb.clue, fb.incorrect_answer) for fb in tb.feedbacks ]}")

    print("\n===== Puzzle Difficulty =====")
    print(final_state.puzzle_difficulty)

if __name__ == '__main__':
    asyncio.run(main())
