import torch
import numpy as np

def insert_embedding_to_batch(learnableToken, sp_indice, batch_p, batch_h, mask_p, mask_h):
    l_p = learnableToken(sp_indice[0])
    mask_p_convert = mask_p.unsqueeze(-1).expand_as(batch_p)

    l_h = learnableToken(sp_indice[1])
    mask_h_convert = mask_h.unsqueeze(-1).expand_as(batch_h)

    return mask_p_convert * l_p + batch_p, mask_h_convert * l_h + batch_h

class DataLoader():
    def __init__(self, data, batch_size, phase, c_ratio, concat, mean_agg_unalign=True, no_unalign=False):
        self.mean_agg_unalign = mean_agg_unalign
        if mean_agg_unalign:
            self.max_length = 32
        else:
            self.max_length = 64
        self.batch_size = batch_size
        self.data = data
        self.length_data = len(data)
        self.phase = phase
        self.c_ratio = c_ratio
        self.concat = concat
        self.pos = 0
        self.no_unalign = no_unalign
        
    def __len__(self):
        return len(self.data)
    
    def epoch_reset(self, shuffle=True):
        self.pos = 0
        if shuffle:
            np.random.shuffle(self.data)
    
    def aggregate(self, embeddings1, embeddings2, pos, indice_c1, indice_c2):
        length1 = len(embeddings1)
        length2 = len(embeddings2)
        if self.no_unalign:
            unmask_length = len(pos)
        else:
            unmask_length = length1 + length2 - len(pos)
        
        if self.concat:
            tensors_p = torch.zeros([self.max_length, 768*2])
            tensors_h = torch.zeros([self.max_length, 768*2])
        else:
            tensors_p = torch.zeros([self.max_length, 768])
            tensors_h = torch.zeros([self.max_length, 768])
        p_token_mask = torch.zeros([self.max_length])
        h_token_mask = torch.zeros([self.max_length])
        mask = torch.zeros([self.max_length])
        mask[:unmask_length] = 1

        i_p = 0
        i_h = 0

        for item in pos:
            helper_tensor_p = torch.zeros([1, length1])
            helper_tensor_h = torch.zeros([1, length2])
            helper_tensor_p[:, item[0]] = 1
            helper_tensor_h[:, item[1]] = 1
            # print(helper_tensor_p.shape, embeddings1.shape, torch.mm(helper_tensor_p, embeddings1).shape)
            tensors_p[i_p, :] = torch.mm(helper_tensor_p, embeddings1)
            tensors_h[i_h, :] = torch.mm(helper_tensor_h, embeddings2)
            i_p += 1
            i_h += 1
        
        if not self.no_unalign:
            for i in indice_c1:
                helper_tensor_p = torch.zeros([1, length1])
                helper_tensor_p[:, i] = 1
                tensors_p[i_p, :] = torch.mm(helper_tensor_p, embeddings1)
                # tensors_h[i_h, :] = self.sp_embeddings(1)
                h_token_mask[i_h] = 1

                i_p += 1
                i_h += 1
            
            for i in indice_c2:
                helper_tensor_h = torch.zeros([1, length2])
                helper_tensor_h[:, i] = 1
                tensors_h[i_h, :] = torch.mm(helper_tensor_h, embeddings2)
                # tensors_p[i_p, :] = self.sp_embeddings(0)
                p_token_mask[i_p] = 1
                
                i_p += 1
                i_h += 1

        return tensors_p, tensors_h, mask, p_token_mask, h_token_mask
    
    def aggregate_mean_unalign(self, embeddings1, embeddings2, pos, indice_c1, indice_c2):
        length1 = len(embeddings1)
        length2 = len(embeddings2)
        has_c1 = 0
        has_c2 = 0
        if not self.no_unalign:
            if len(indice_c1) > 0:
                has_c1 = 1
            if len(indice_c2) > 0:
                has_c2 = 1
        unmask_length = len(pos) + has_c1 + has_c2

        if self.concat:
            tensors_p = torch.zeros([self.max_length, 768*2])
            tensors_h = torch.zeros([self.max_length, 768*2])
        else:
            tensors_p = torch.zeros([self.max_length, 768])
            tensors_h = torch.zeros([self.max_length, 768])
        p_token_mask = torch.zeros([self.max_length])
        h_token_mask = torch.zeros([self.max_length])
        mask = torch.zeros([self.max_length])
        mask[:unmask_length] = 1

        i_p = 0
        i_h = 0
        for item in pos:
            helper_tensor_p = torch.zeros([1, length1])
            helper_tensor_h = torch.zeros([1, length2])
            helper_tensor_p[:, item[0]] = 1
            helper_tensor_h[:, item[1]] = 1
            # print(helper_tensor_p.shape, embeddings1.shape, torch.mm(helper_tensor_p, embeddings1).shape)
            tensors_p[i_p, :] = torch.mm(helper_tensor_p, embeddings1)
            tensors_h[i_h, :] = torch.mm(helper_tensor_h, embeddings2)
            i_p += 1
            i_h += 1
        
        if not self.no_unalign:
            if len(indice_c1) != 0:
                mean_unaligned_p = []
                for i in indice_c1:
                    helper_tensor_p = torch.zeros([1, length1])
                    helper_tensor_p[:, i] = 1
                    mean_unaligned_p.append(torch.mm(helper_tensor_p, embeddings1))
                mean_unaligned_p = torch.stack(mean_unaligned_p).mean(dim=0)
                tensors_p[i_p, :] = mean_unaligned_p
                h_token_mask[i_h] = 1
                i_p += 1
                i_h += 1

            if len(indice_c2) != 0:
                mean_unaligned_h = []
                for i in indice_c2:
                    helper_tensor_h = torch.zeros([1, length2])
                    helper_tensor_h[:, i] = 1
                    mean_unaligned_h.append(torch.mm(helper_tensor_h, embeddings2))
                mean_unaligned_h = torch.stack(mean_unaligned_h).mean(dim=0)
                tensors_h[i_h, :] = mean_unaligned_h
                p_token_mask[i_p] = 1

        return tensors_p, tensors_h, mask, p_token_mask, h_token_mask

    def get_batch(self):
        self.epoch_reset(shuffle=True if self.phase == 'train' else False)

        while self.pos < self.length_data:
            
            aligned_text = []
            result_tensors_p = []
            result_tensors_h = []
            masks = []
            p_token_masks = []
            h_token_masks = []
            labels = []
            indice = []
            for i in range(self.batch_size):
                item = self.data[self.pos]
                this_id = item['id']
                c_p = item['c_p']
                c_h = item['c_h']
                indice_c1 = item['p_not_aligned']
                indice_c2 = item['h_not_aligned']
                pos = item['p_h_aligned']

                # ---------------------------------------------------------------------------
                # random aligner
                # np.random.seed(i)
                # aligned_length = len(pos)
                # indice_c1 = np.arange(0, len(c_p)).tolist()
                # indice_c2 = np.arange(0, len(c_h)).tolist()
                # np.random.seed(i)
                # a = np.random.choice(indice_c1, size=aligned_length, replace=False)
                # np.random.seed(i)
                # b = np.random.choice(indice_c2, size=aligned_length, replace=False)
                
                # pos = []
                # for i in range(aligned_length):
                #     pos.append((a[i], b[i]))

                # for ii in pos:
                #     indice_c1.remove(ii[0])
                #     indice_c2.remove(ii[1])
                # ---------------------------------------------------------------------------

                if self.phase == 'test':
                    aligned = []
                    p_unaligned = []
                    h_unaligned = []
                    for ii in pos:
                        aligned.append((c_p[ii[0]],c_h[ii[1]]))
                    for ii in indice_c1:
                        p_unaligned.append(c_p[ii])
                    for ii in indice_c2:
                        h_unaligned.append(c_h[ii])
                    indice.append(this_id)
                    aligned_text.append([aligned, p_unaligned, h_unaligned])

                if self.concat:
                    embeddings1 = torch.cat((torch.from_numpy(item['ccem1']), torch.from_numpy(item['cem1'])), -1)
                    embeddings2 = torch.cat((torch.from_numpy(item['ccem2']), torch.from_numpy(item['cem2'])), -1)
                else:
                    if self.c_ratio == 1:
                        embeddings1 = torch.from_numpy(item['ccem1'])
                        embeddings2 = torch.from_numpy(item['ccem2'])
                    elif self.c_ratio == 0:
                        embeddings1 = torch.from_numpy(item['cem1'])
                        embeddings2 = torch.from_numpy(item['cem2'])
                    else:
                        embeddings1 = torch.from_numpy(item['ccem1']) * self.c_ratio + torch.from_numpy(item['cem1']) * (1-self.c_ratio)
                        embeddings2 = torch.from_numpy(item['ccem2']) * self.c_ratio + torch.from_numpy(item['cem2']) * (1-self.c_ratio)
                label = item['label']
                
                if self.mean_agg_unalign:
                    tensors_p, tensors_h, mask, p_token_mask, h_token_mask = self.aggregate_mean_unalign(embeddings1, embeddings2, pos, indice_c1, indice_c2)
                else:
                    tensors_p, tensors_h, mask, p_token_mask, h_token_mask = self.aggregate(embeddings1, embeddings2, pos, indice_c1, indice_c2)
                
                result_tensors_p.append(tensors_p)
                result_tensors_h.append(tensors_h)
                masks.append(mask)
                p_token_masks.append(p_token_mask)
                h_token_masks.append(h_token_mask)
                labels.append(torch.LongTensor([label]))
                # labels.append(torch.LongTensor([label]) * torch.ones([self.max_length]).long())

                self.pos += 1
                if self.pos >= self.length_data:
                    break

            yield torch.stack(result_tensors_p), torch.stack(result_tensors_h), torch.stack(masks), torch.stack(p_token_masks), torch.stack(h_token_masks), torch.cat(labels), aligned_text, indice
            # yield torch.stack(result_tensors_p), torch.stack(result_tensors_h), torch.stack(masks), torch.stack(p_token_masks), torch.stack(h_token_masks), torch.stack(labels), aligned_text, indice







