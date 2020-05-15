"""
Insert the misinformation JSONL to a Google Sheet
"""
import argparse
from pathlib import Path
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from backend.ml.misconception import MisconceptionDataset


class MisconceptionDatasetToGSheets:

    def __init__(self):
        # If modifying these scopes, delete the file token.pickle.
        self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

        # The ID and range of a sample spreadsheet.
        self.SPREADSHEET_ID = '1ujDu_vVqSP3pfbZvkrtpZBNnwWt-1kDIDUvzOnVY5uc'
        self.MAX_SOURCES = 5
        self.MAX_VARIATIONS = 5

        self.creds = None
        self.service = None

    def start_service(self):
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if Path('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)

        self.service = build('sheets', 'v4', credentials=self.creds)
        # Call the Sheets API
        self.sheets = self.service.spreadsheets()

    def write_dataset(self, misconceptions: MisconceptionDataset,
                      range_name: str) -> None:
        values = []

        # Header
        row = ["Id", "Canonical Sentence", "Origin", "Reliability", "Category"]
        for j in range(self.MAX_SOURCES):
            row.append("source" + str(j+1))
        for j in range(self.MAX_VARIATIONS):
            row.append("pos_variant" + str(j+1))
        for j in range(self.MAX_VARIATIONS):
            row.append("neg_variant" + str(j+1))
        values.append(row)

        for i in range(len(misconceptions)):
            inst = misconceptions[i]
            row = [inst.id, inst.canonical_sentence, inst.origin, inst.reliability_score, inst.category[0]]
            for j in range(self.MAX_SOURCES):
                if j < len(inst.sources):
                    row.append(inst.sources[j])
                else:
                    row.append("")
            for j in range(self.MAX_VARIATIONS):
                if j < len(inst.pos_variations):
                    row.append(inst.pos_variations[j])
                else:
                    row.append("")
            for j in range(self.MAX_VARIATIONS):
                if j < len(inst.neg_variations):
                    row.append(inst.neg_variations[j])
                else:
                    row.append("")
            values.append(row)

        body = {
            'values': values
        }
        result = self.sheets.values().update(spreadsheetId=self.SPREADSHEET_ID, range=range_name,
                                             valueInputOption="USER_ENTERED", body=body).execute()
        print('{0} cells updated.'.format(result.get('updatedCells')))

    def read_dataset(self):
        return NotImplementedError
        # result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
        #                             range=SAMPLE_RANGE_NAME).execute()
        # values = result.get('values', [])

        # if not values:
        #     print('No data found.')
        # else:
        #     for row in values:
        #         print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_file', type=Path, default=Path('misconceptions.jsonl'))
    parser.add_argument('-r', '--range_name', type=str, default='Wikipedia!A1')
    args = parser.parse_args()

    with open(args.dataset_file, 'r') as f:
        misconceptions = MisconceptionDataset.from_jsonl(f)

    obj = MisconceptionDatasetToGSheets()
    obj.start_service()
    obj.write_dataset(misconceptions, args.range_name)
