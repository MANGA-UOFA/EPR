import spacy
import numpy as np
import random

class Chunker:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)
        self.doc = None

    def convert_to_spacy(self, string):
        self.doc = self.nlp(string)
        # for token in self.doc:
        #     print(token.text, token.pos_, token.tag_)
        
    def noun_chunk(self):
        nchunks = list(self.doc.noun_chunks)
        span_pos = []
        for chunk in nchunks:
            span_pos.append([chunk.start, chunk.end])

        return nchunks, span_pos
    
    def after_prep_chunk(self):
        nchunks, spans_pos = self.noun_chunk()
        pchunks = []
        for i in range(len(nchunks)):
            index = spans_pos[i][0] - 1
            if index < 0: # nothing in front of this chunk
                pchunks.append(nchunks[i])
            else:
                left_token = self.doc[index]
                if left_token.tag_ == 'IN': # define prep chunk is ADP + noun chunk
                    pchunks.append(self.doc[index:nchunks[i].end])
                    spans_pos[i][0] = index
                else:
                    pchunks.append(nchunks[i])
        return pchunks, spans_pos
    
    def extract_token_attr(self, attr_string=['VERB', 'ADV', 'ADJ', 'INTJ', 'NOUN', 'PROPN']): 
        # Universal POS tags
        # we prefer open words
        # https://universaldependencies.org/u/pos/
        out = []
        for token in self.doc:
            # print(token, token.pos_)
            if token.pos_ in attr_string:
                out.append(token)
        # out.sort(key=lambda x: x.i)
        return out
    
    def extract_verb_phrase(self):
        # aux + not + verb + (rb)
        # verb + rb
        out = []
        spans = []
        for i in range(len(self.doc)):
            if self.doc[i].pos_ == 'VERB':
                if i-2 >= 0 and self.doc[i-1].pos_ == 'PART' and self.doc[i-2].pos_ == 'AUX':
                    if i+1 < len(self.doc) and self.doc[i+1].tag_ == 'RP':
                        out.append(self.doc[i-2:i+2])
                        # spans.append((i-1, i+2))
                    else:
                        out.append(self.doc[i-2:i+1])
                        # spans.append((i-1, i+1))
                else:
                    if i+1 < len(self.doc) and self.doc[i+1].tag_ == 'RP':
                        out.append(self.doc[i:i+2])
                        # spans.append((i, i+2))
        # out.sort(key=lambda x: x[0].i)
        for o in out:
            spans.append([o.start, o.end])
        return out, spans
        
    def process(self, string):
        self.convert_to_spacy(string)

        chunks, span_pos = self.after_prep_chunk()
        noun_chunk_indice = []
        for span in span_pos:
            this_range = np.arange(span[0], span[1]).tolist()
            noun_chunk_indice += this_range

        vp_indice = []
        vp_new_span = []
        vps_new = []
        vps, spans_vp = self.extract_verb_phrase()
        for item in zip(vps, spans_vp):
            vp = item[0]
            span = item[1]
            this_range = np.arange(span[0], span[1]).tolist()
            if len(set(this_range) & set(noun_chunk_indice)) == 0:
                vp_indice += this_range
                vp_new_span.append(span)
                vps_new.append(vp)

        phrase_indice = vp_indice + noun_chunk_indice

        tokens = self.extract_token_attr()
        filtered_tokens = []
        for t in tokens:
            if t.i not in phrase_indice:
                filtered_tokens.append(t)

        aDict = {}
        blist = []
        for i, sen in enumerate(chunks):
            aDict[sen] = span_pos[i][0]
            blist.append((sen.start_char, sen.end_char))
        for i, sen in enumerate(vps_new):
            aDict[sen] = vp_new_span[i][0]
            blist.append((sen.start_char, sen.end_char))
        for i, w in enumerate(filtered_tokens):
            aDict[w] = w.i
            blist.append((w.idx, w.idx+len(w)))
        
        sortedDict = sorted(aDict.items(), key=lambda item: item[1])
        final_chunks_text = [s[0].text for s in sortedDict]
        blist.sort(key=lambda tup: tup[0])

        return final_chunks_text, blist

class RandomChunker:
    def __init__(self):
        self.spacy_chunker = Chunker()

    def constrained_sum_sample_pos(self, n, total):
        # ref: https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value
        """Return a randomly chosen list of n positive integers summing to total.
        Each such list is equally likely to occur."""

        dividers = sorted(random.sample(range(1, total), n - 1))
        return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    def process(self, string):
        chunks_text = []
        blist = []
        final_chunks_text, _ = self.spacy_chunker.process(string)
        if len(final_chunks_text) == 0:
            return chunks_text, blist
        doc = self.spacy_chunker.doc
        indice = self.constrained_sum_sample_pos(len(final_chunks_text), len(doc))
        i = 0
        for length in indice:
            this_chunk = doc[i:i+length]
            chunks_text.append(this_chunk.text)
            blist.append((this_chunk.start_char, this_chunk.end_char))
            i += length

        return chunks_text, blist

if __name__ == '__main__':
    c = RandomChunker()
    c1 = c.process('There is no boy playing outdoors and there is no man smiling')
    # c1 = c.process('man, man')
    print(c1)