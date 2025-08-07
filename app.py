import json
import os
import logging
import asyncio
from typing import Optional, Tuple, Dict
import openai
import difflib
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_BOT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CREDS_JSON = os.getenv("SHEETS_CREDS")
CUSTOMERS_JSON = os.getenv("CUSTOMERS_JSON")
SHEET_NAME = "9_ტონა_ფული"
WORKSHEET_NAME = "Payments"

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not CREDS_JSON or not CUSTOMERS_JSON:
    raise ValueError("Environment variables must be set")

try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(CREDS_JSON), SCOPE)
CLIENT = gspread.authorize(CREDS)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class PaymentBot:
    def __init__(self) -> None:
        self.customers: list[str] = []
        self.name_to_full: Dict[str, str] = {}
        self._load_customers()

    def _load_customers(self) -> None:
        try:
            self.customers = json.loads(CUSTOMERS_JSON)
            logger.info(f"Loaded {len(self.customers)} customers")
        except Exception as e:
            self.customers = []
            logger.warning(f"Failed to load customers: {e}")
        
        for customer in self.customers:
            customer = customer.strip()
            if customer:
                match = re.match(r'\((.*?)\)\s*(.*)', customer)
                if match:
                    name = match.group(2).strip()
                    self.name_to_full[name] = customer
                else:
                    self.name_to_full[customer] = customer

    def parse_payment(self, text: str) -> Optional[Tuple[str, float]]:
        pattern = r'^(.*)\s+(\d+(?:\.\d+)?)\s*(?:GEL|USD|EUR)?$'
        match = re.match(pattern, text.strip())
        
        if match:
            name = match.group(1).strip()
            try:
                amount = float(match.group(2))
                if amount > 0:
                    return name, amount
            except ValueError:
                pass
        return None

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message or update.edited_message
        if not message or not message.text:
            return

        text = message.text
        username = message.from_user.username or f"{message.from_user.first_name or ''} {message.from_user.last_name or ''}".strip()
        source = 'Edited' if update.edited_message else 'Direct'

        logger.info(f"Processing message from {username}: '{text}'")

        parsed = self.parse_payment(text)
        if not parsed:
            logger.debug(f"Could not parse payment from: '{text}'")
            return

        name, amount = parsed
        logger.info(f"Parsed payment: {name} -> {amount}")

        customer_full = await self.find_customer(name)
        
        if customer_full:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            success = await self.record_to_sheets(timestamp, customer_full, str(amount), source, username)
            
            if success:
                await message.reply_text(f"✅ გადახდა ჩაწერილია:\n{customer_full}\nთანხა: {amount} ₾")
                logger.info(f"Payment recorded: {customer_full} {amount} by {username}")
            else:
                await message.reply_text("❌ ვერ მოხერხდა ჩაწერა Google Sheets-ში. გთხოვთ სცადოთ მოგვიანებით.")
        else:
            await message.reply_text(f"❌ კლიენტი '{name}' ვერ მოიძებნა.\nგთხოვთ გადაამოწმოთ სახელი და სცადოთ თავიდან.")
            logger.warning(f"Customer not found: '{name}'")

    async def find_customer(self, name: str) -> Optional[str]:
        if name in self.name_to_full:
            logger.info(f"Direct match found: '{name}'")
            return self.name_to_full[name]
        
        if name in self.customers:
            logger.info(f"Full customer string provided: '{name}'")
            return name
        
        name_lower = name.lower()
        for short_name, full_name in self.name_to_full.items():
            if short_name.lower() == name_lower:
                logger.info(f"Case-insensitive match found: '{name}' -> '{full_name}'")
                return full_name
        
        logger.info(f"No direct match for '{name}', trying GPT mapping...")
        gpt_result = await self.map_customer_with_gpt(name)
        
        if gpt_result:
            logger.info(f"GPT successfully mapped: '{name}' -> '{gpt_result}'")
            return gpt_result
        
        logger.warning(f"Could not find customer: '{name}'")
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def map_customer_with_gpt(self, customer_name: str) -> Optional[str]:
        if not self.customers:
            return None
        
        customer_names = list(self.name_to_full.keys())
        closest_matches = difflib.get_close_matches(customer_name, customer_names, n=20, cutoff=0.2)
        
        if closest_matches:
            relevant_customers = [self.name_to_full[name] for name in closest_matches]
        else:
            relevant_customers = self.customers[:30]
        
        system_prompt = (
            "You are a customer name mapping assistant. Map the input name to the EXACT customer from the list.\n"
            "Handle typos, abbreviations, and variations.\n"
            "Return ONLY the exact customer string from the list, or 'null' if no match.\n\n"
            f"CUSTOMERS:\n{json.dumps(relevant_customers, ensure_ascii=False)}"
        )
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Find customer: {customer_name}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            if result != "null" and result in self.customers:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"GPT mapping error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def record_to_sheets(self, timestamp: str, customer: str, amount: str, source: str, sender: str) -> bool:
        try:
            sheet = CLIENT.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
            row = [timestamp, customer, amount, source, sender]
            sheet.append_row(row)
            logger.info(f"Recorded to Sheets: {row}")
            return True
        except Exception as e:
            logger.error(f"Error recording to Sheets: {e}")
            return False

bot = PaymentBot()
application = Application.builder().token(TELEGRAM_TOKEN).build()

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_message(update, context)

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

if __name__ == '__main__':
    logger.info("Starting Payment Bot polling...")
    asyncio.run(application.run_polling(timeout=10))