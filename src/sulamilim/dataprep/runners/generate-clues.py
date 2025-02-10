import argparse
import asyncio
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from globalog import LOG
from jserpy import serialize_json_as_dict
# Example: Using a 'langchain agentic flow' style to generate word ladder clues
# for a CrossClimb-style puzzle in Hebrew.

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm

from sulamilim.dataprep.scrapping.fetch import fetch_content
from sulamilim.dataprep.scrapping.llm_reader import parse_for_llm


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
    "צור רמז של משפט קצר אחד שמתאר או רומז למילה. "
    "נסה להיות יצירתי, במקרים של דו משמעות תוכל להתייחס לאספקט אחד של המילה או כמה, תוכל לתת ביטוי עם חלק חסר "
    "כאשר המילמה משלימה את החלק החסר בביטוי."
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
    "צור רמז של משפט קצר אחד, יכול להיות עם שני חלקים שמתאר או רומז לזוג המילים. "
    "נסה להיות יצירתי, במקרים של דו משמעות תוכל להתייחס לאספקט אחד של המילה או כמה, תוכל לתת ביטוי עם חלק חסר "
    "כאשר המילה משלימה את החלק החסר בביטוי."
    "\n\nדוגמאות לרמזים:"
    f"{_TOP_BOTTOM_HEBREW_EXAMPLES}"
    "\n\nועד דוגמאות מאנגלית בשביל רעיונות"
    f"{_TOP_BOTTOM_WORDS_CLUE_EXAMPLES}"
)


# Instantiate LLMs
llm_clue_gen = ChatOpenAI(model='o1')#, temperature=0.7)  # Generates clues in Hebrew
llm_solver = ChatOpenAI(model='o1')#, temperature=0.5)     # Solves puzzle in Hebrew
llm_evaluator = ChatOpenAI(model='o1')#, temperature=0.3)  # Evaluates clue difficulty

@dataclass
class Rung:
    word: str
    clue: str = ''
    solved_answer: str = ''
    difficulty_evaluation: str = ''
    definition_text: str = ''  # store fetched text for reference
    answer_explanation: str = ''

@dataclass
class TopBottomRung:
    top_word: str
    bottom_word: str
    top_definition_text: str = ''
    bottom_definition_text: str = ''
    clue: str = ''
    solved_top: str = ''
    solved_bottom: str = ''
    difficulty_evaluation: str = ''

@dataclass
class PuzzleState:
    rung_words: List[Rung] = field(default_factory=list)
    top_bottom_rungs: Optional[TopBottomRung] = None
    structured_output: Dict[str, str | dict[str, str]] = field(default_factory=dict)

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

def generate_rung_clues(state: PuzzleState) -> PuzzleState:
    """
    For each rung word, fetch the definition page as markdown, then generate a Hebrew clue.
    """
    system_msg = SystemMessage(content=_SYSTEM_PROMPT_WORD_CLUES)

    for rung in state.rung_words:
        # Build the prompt to generate a clue in Hebrew
        content_text = (
            f"המילה: {rung.word}\n\n"
            f"להלן תוכן (בפורמט Markdown) מתוך האתר:\n\n"
            f"{rung.definition_text}\n\n"
            "צור רמז של משפט אחד קצר בעברית המסייע לנחש את המילה."
        )
        LOG.info(f"[Generate-clues] send request to LLM for word \"{rung.word}\" with prompt of size {len(content_text)}")
        response = llm_clue_gen.invoke([system_msg, HumanMessage(content=content_text)])
        rung.clue = response.content.strip()
        LOG.info(f"[Generate-clues] Got response: {rung.clue}")

    return state

def generate_top_bottom_clue(state: PuzzleState) -> PuzzleState:
    """
    Generate a single short clue in Hebrew referencing both top and bottom words.
    """
    system_msg = SystemMessage(content=_SYSTEM_PROMPT_TOP_BOTTOM_CLUES)
    tb = state.top_bottom_rungs
    prompt = (
        f"מילה עליונה: {tb.top_word}\n"
        f"להלן תוכן המילה העליונה (בפורמט Markdown) מתוך האתר:\n\n"
        f"{tb.top_definition_text}\n\n"
        f"מילה תחתונה: {tb.bottom_word}\n\n"
        f"להלן תוכן המילה התחתונה (בפורמט Markdown) מתוך האתר:\n\n"
        f"{tb.bottom_definition_text}\n\n"
        "צור רמז מקוצר בעברית המאגד את שתיהן בהקשר משותף."
    )
    LOG.info(f"[Generate-clues] send request to LLM for top word: {tb.top_word} and bottom word: {tb.bottom_word} with prompt of size {len(prompt)}")
    response = llm_clue_gen.invoke([system_msg, HumanMessage(content=prompt)])
    state.top_bottom_rungs.clue = response.content.strip()
    LOG.info(f"[Generate-top-bottom-clues] Got response: {state.top_bottom_rungs.clue}")
    return state

def solve_word_clues(state: PuzzleState) -> PuzzleState:
    """
    Let the solver LLM see all the rung clues (and top/bottom if needed)
    along with instructions, and ask it to solve all in one shot.
    """
    # Build a single message that has all rung clues
    solver_instructions = (
        "אתה מקבל רשימת רמזים בעברית עבור סולם מילים (word ladder). "
        "המשימה שלך היא לזהות את המילים המתאימות לכל רמז, ולסדר את המילים לפי סדר, בהתחשב בכך "
        "שכל מילה שונה רק באות אחת מהמילה הבאה. כמובן שגם זה רמז למילים אפשריות היכולות לשמש כתשובה\n\n"
        "הרמזים:\n"
    )

    clues_order = list(range(len(state.rung_words)))
    random.shuffle(clues_order)
    for i, rung_index in enumerate(clues_order):
        solver_instructions += f"{i+1}. רמז: {state.rung_words[i].clue}\n"

    solver_instructions += ("\nספק את הפתרונות למילים, וסדר אותן לפי סדר. תן את הערכתך הטובה ביותר."
                            "בתשובתך שים כל מילה בשורה נפרדת, וסדר את המילים לפי סדר (עולה או יורד הכיוון לא חשוב)  "
                            "העיקר שכל מילה תיבדל באות אחת מהמילים לצידה, "
                            "ולצד כל מילה רשום את מספר המילה המקורי (לפני הסידור) כפי שנכתב בהוראות האלה."
                            "בכל שורה הפרד את המילה מההסבר בעזרת נקודותיים (:) כמו בתבנית הבאה: {i}. {word}: {explanation}"
                            " כאשר i הוא מספר המילה. לדוגמה: ֿ\"1. מילה: הסבר מדוע זיהינו את המילה הזו\"")

    LOG.info(f"[Solve-words] Sending prompt to LLM {llm_solver.name}: instructions: {solver_instructions}")
    response = llm_solver.invoke([HumanMessage(content=solver_instructions)])
    # Try to parse response as best as possible
    # We'll assume the solver tries to provide them in order or labeled by rung
    # We'll just store the entire text for each rung for reference
    solutions_text = response.content.strip()
    LOG.info(f"[Solve Words] Got response, answer: {solutions_text}")

    # For a more advanced approach, you could attempt to parse the solutions_text using regex or an internal parser.
    # For now, we just store it in 'solved_answer' for each rung.
    # Or we can store it in a single string, but let's distribute it in a naive manner:
    lines = solutions_text.splitlines()
    # We attempt to match lines that might contain the rung # and solution
    for rung_index in range(len(clues_order)):
        for line in lines:
            # Check if line has something like "1." or "1) " etc.
            # naive approach: check f"{rung_idx+1}" substring
            if str(rung_index+1) in line:
                split_line = line.split(':', maxsplit=1)
                word = split_line[0].strip()
                word = ''.join(filter(str.isalpha, word))
                state.rung_words[rung_index].solved_answer = word
                if len(split_line) == 2:
                    explanation = split_line[1].strip()
                    state.rung_words[rung_index].answer_explanation = explanation

    return state

def solve_top_bottom_clue(state: PuzzleState) -> PuzzleState:
    solver_instructions = ("לפניך חידה בה צריך לגלות שתי מילים שונות. מילה ראשונה ומילה שניה."
                           "על מנת לגלות את שתי המילים מובאים שני סוגי אינפורמציה:"
                           "\n- רמז משותף לשתי המילים שעוזר לפתור את שתיהן."
                           "\n- לצד כל מילה יש מילה נוספת (לא בהכרח קשורה) הנבדלת ממנה באות אחת בלבד."
                           "\n\nלהלן הרמז המשותף:"
                           f"  {state.top_bottom_rungs.clue}"
                           f"\n\nלהלן המילה הנוספת לצד כל אחת מהתשובות - מילת התשובה צריכה להבידל באות אחת מהמילה שלצידה:"
                           f"\n1. {state.rung_words[0].word}"
                           f"\n2. {state.rung_words[-1].word}"
                           "\n\n בתשובתך כתוב כל מילה שגילית בשורה נפרדת, ועבור כל מילה ציין את מספר המילה (1. או 2.) לפני המילה "
                           )

    LOG.info(f"[Solve-top-bottom] Sending prompt to LLM {llm_solver.name}: instructions: {solver_instructions}")
    response = llm_solver.invoke([HumanMessage(content=solver_instructions)])
    solutions_text = response.content.strip()
    LOG.info(f"[Solve-top-bottom] Got response: answer: {solutions_text}")
    lines = solutions_text.splitlines()
    answers = ['', '']
    for rung_index in (1, 2):
        for line in lines:
            if str(rung_index) in line:
                word = ''.join(filter(str.isalpha, line))
                answers[rung_index-1] = word

    state.top_bottom_rungs.solved_top = answers[0]
    state.top_bottom_rungs.solved_bottom = answers[1]
    return state


def evaluate_clues_difficulty(state: PuzzleState) -> PuzzleState:
    system_msg = SystemMessage(content=(
        "אתה צריך להעריך את רמת הקושי (קל, בינוני, או קשה) של הרמז בעברית. "
        "ענה במילה אחת בלבד."
    ))
    for rung in state.rung_words:
        prompt = f"רמז: {rung.clue}\nהמילה היא: {rung.word}\nמה רמת הקושי?"
        response = llm_evaluator.invoke([system_msg, HumanMessage(content=prompt)])
        rung.difficulty_evaluation = response.content.strip()
    return state

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
            "difficulty": rung.difficulty_evaluation
        }

    state.structured_output['top_bottom'] = {
        'top_word': state.top_bottom_rungs.top_word,
        'bottom_word': state.top_bottom_rungs.bottom_word,
        'clue': state.top_bottom_rungs.clue,
        'solved_top': state.top_bottom_rungs.solved_top,
        'solved_bottom': state.top_bottom_rungs.solved_bottom,
        'difficulty': state.top_bottom_rungs.difficulty_evaluation
    }
    return state


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

    # Build the agentic flow
    workflow = StateGraph(PuzzleState)
    workflow.add_node('generate_rung_clues', generate_rung_clues)
    workflow.add_node('generate_top_bottom_clue', generate_top_bottom_clue)
    workflow.add_node('solve_words', solve_word_clues)
    workflow.add_node('solve_top_bottom', solve_top_bottom_clue)
    # workflow.add_node('evaluate_clues_difficulty', evaluate_clues_difficulty)
    workflow.add_node('prepare_structured_output', prepare_structured_output)

    workflow.add_edge('generate_rung_clues', 'generate_top_bottom_clue')
    workflow.add_edge('generate_top_bottom_clue', 'solve_words')
    workflow.add_edge('solve_words', 'solve_top_bottom')
    # workflow.add_edge('solve_top_bottom', 'evaluate_clues_difficulty')
    # workflow.add_edge('evaluate_clues_difficulty', 'prepare_structured_output')
    workflow.add_edge('solve_top_bottom', 'prepare_structured_output')

    workflow.set_entry_point('generate_rung_clues')
    workflow.set_finish_point('prepare_structured_output')

    # Compile and run
    app = workflow.compile()
    final_state = app.invoke(puzzle_state)

    print(final_state['structured_output'])
    out = json.dumps(final_state['structured_output'], indent=2)
    print(out)


if __name__ == '__main__':
    asyncio.run(main())
