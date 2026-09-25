#!/usr/bin/env python3
"""Resolve every DOI through Crossref and reject preprint records."""
from __future__ import annotations
import argparse, json, re, urllib.request
from pathlib import Path


def esc(s: str) -> str:
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def bib_from_crossref(key: str, msg: dict) -> str:
    kind = 'article' if msg['type'] == 'journal-article' else 'inproceedings'
    fields = {'title':'{' + esc(msg['title'][0]) + '}',
              'author':' and '.join(esc(a.get('family','')+', '+a.get('given','')) for a in msg.get('author',[])),
              'year':str(msg.get('published',msg.get('issued'))['date-parts'][0][0]),
              'doi':msg['DOI'], 'url':'https://doi.org/'+msg['DOI']}
    if kind=='article': fields['journal']=esc(msg.get('container-title',[''])[0])
    else: fields['booktitle']=esc(msg.get('container-title',[''])[0])
    for k in ('volume','issue','page','publisher'):
        if msg.get(k): fields[k]=esc(str(msg[k]))
    return '@'+kind+'{'+key+',\n'+''.join('  '+k+' = {'+v+'},\n' for k,v in fields.items())+'}\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    records=json.loads(args.catalog.read_text());out=[];seen=set()
    for r in records:
        key=r['key'];assert key not in seen;seen.add(key)
        if 'doi' in r:
            req=urllib.request.Request('https://api.crossref.org/works/'+r['doi'],headers={'User-Agent':'proof-economic-agents-reference-check/1.0 (scholarly metadata validation)'})
            with urllib.request.urlopen(req, timeout=30) as f: msg=json.load(f)['message']
            if msg['DOI'].lower()!=r['doi'].lower() or msg['type'] in {'posted-content','preprint'}: raise ValueError(f'wrong/preprint DOI: {key}')
            if r['title_contains'].casefold() not in msg['title'][0].casefold(): raise ValueError(f'title mismatch: {key}')
            out.append(bib_from_crossref(key,msg))
            print(key,msg['DOI'],msg['type'],msg['title'][0])
        else:
            if not ('venue_exception' in r or 'input_exception' in r): raise ValueError(f'unjustified manual citation: {key}')
            bib=r['bibtex']
            if not re.search(r'@\w+\{'+re.escape(key)+r',',bib): raise ValueError(f'key mismatch: {key}')
            out.append(bib+'\n')
            print(key,'explicit exception:',r.get('venue_exception',r.get('input_exception')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(out))

if __name__=='__main__': main()
