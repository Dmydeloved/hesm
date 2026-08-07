import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.extractor import TopicExtractor


if __name__ == "__main__":
    topic_extractor = TopicExtractor()

    single_topic_input = "Where did Caroline move from 4 years ago?"
    single_topic_result = topic_extractor.extract(user_input=single_topic_input)
    print("single topic result:")
    print(json.dumps(single_topic_result, ensure_ascii=False, indent=4))

    # multi_topic_input = (
    #     "i need an expensive restaurant in the center, and i also need a hotel "
    #     "near cambridge station"
    # )
    # multi_topic_result = topic_extractor.extract(user_input=multi_topic_input)
    # print("multi topic result:")
    # print(json.dumps(multi_topic_result, ensure_ascii=False, indent=4))
