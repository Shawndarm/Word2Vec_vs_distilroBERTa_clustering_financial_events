#################### Main Functions for data extraction ####################
import pandas as pd
from newspaper import Article
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()
import asyncio
import re
from datetime import datetime, timezone
from telethon import TelegramClient


###################### News extraction ######################


def check_density(path):
    """Loads dataset, checks for date coverage and identifies gaps."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # Identify gaps in time series
    full_range = pd.date_range(df["date"].min(), df["date"].max())
    missing = full_range.difference(df["date"].unique())
    # Print summary statistics
    print(f"Total records: {len(df)}")
    print(f"Coverage: {df['date'].nunique()} / {len(full_range)} days")
    print(f"Gaps found: {len(missing)} days")

    return df


def scrape_content(df, filename):
    """Scrapes full text and saves valid results incrementally."""
    results = []
    print("Starting extraction process...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            art = Article(row["url"])
            art.download()
            art.parse()
            # Filter for high-quality content only (>500 chars)
            if len(art.text) > 500:
                results.append(
                    {
                        "date": row["date"],
                        "headline": art.title,
                        "body": art.text,
                        "url": row["url"],
                        "source": row["source"],
                    }
                )
        except:
            continue
        # Incremental save every 100 successful extractions
        if len(results) % 100 == 0 and results:
            pd.DataFrame(results).to_csv(filename, index=False)
    # Final save
    final_df = pd.DataFrame(results)
    final_df.to_csv(filename, index=False)
    print(f"Complete: {len(final_df)} valid articles extracted.")
    return final_df


###################### Tweets extraction ######################

import nest_asyncio

nest_asyncio.apply()


# REGEX
NASDAQ_RE = re.compile(
    r"\b(nasdaq|qqq|ndx|nq|nas100|nasdaq100|nvda|aapl|msft|"
    r"googl|amzn|meta|tsla|faang|mag7|buy|sell|bullish|bearish|long|short)\b",
    re.IGNORECASE,
)


# remove emojis
def clean_text(text):
    return re.sub(r"[^\x00-\x7F]+", " ", text)


# Main function
def scrape_telegram_nasdaq(API_ID, API_HASH, PHONE, output_path):

    CHANNELS = [
        "nas100masters",
        "nas100_trading_signal",
        "nasdaq_freesignals",
        "EagleSpiritforexacademyNASDAQ100",
        "FREEUS30XAUUSDSIGNALS",
        "Nas100_Trading_forex_signal",
        "financialjuice",
        "TradingViewIdeas",
        "investing_com",
        "forexsignalsfactory",
        "wallstreetmemes",
        "marketsignals",
        "vanillafinancenews",
        "FREEDOMFINANCE",
        "money",
        "finance",
        "curvefi",
        "personalfinancesg",
        "finance1",
        "SerezhaCalls",
        "nasdaq_r",
        "stocks",
        "x_stocks_bot",
        "stocks_0",
    ]

    START_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
    END_DATE = datetime(2026, 4, 1, tzinfo=timezone.utc)

    async def collect():
        client = TelegramClient("session", API_ID, API_HASH)
        await client.start(phone=PHONE)

        all_posts = []

        for channel in CHANNELS:
            try:
                entity = await client.get_entity(channel)

                async for msg in client.iter_messages(entity, offset_date=END_DATE):
                    if msg.date < START_DATE:
                        break

                    if not msg.text:
                        continue

                    if not NASDAQ_RE.search(msg.text):
                        continue

                    all_posts.append(
                        {
                            "id": f"{channel}_{msg.id}",
                            "channel": channel,
                            "date": msg.date.strftime("%Y-%m-%d"),
                            "text": clean_text(msg.text)[:520],
                            "views": getattr(msg, "views", 0) or 0,
                        }
                    )

            except Exception as e:
                print(f"Error with {channel}: {e}")

        await client.disconnect()
        return all_posts

    # run async
    posts = asyncio.run(collect())

    # dataframe
    if not posts:
        print("No data collected")
        return pd.DataFrame()

    df = pd.DataFrame(posts).drop_duplicates("id")
    # sauvegarde CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Collected {len(df)} messages")
    print(f"Days covered: {df['date'].nunique()}")
    print(f"Saved to: {output_path}")

    return df
