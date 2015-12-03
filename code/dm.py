from collections import defaultdict
import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import preprocessing

categories = ['office equipment', 'jewelry', 'mobile', 'computers', 'auto parts', 'home goods', 'sporting goods', 'books and music', 'clothing', 'merchandise', 'furniture']

def category_features(max_category):
	return [int(max_category == category) for category in categories]
	
def get_labels(user_file):
	f = open(user_file)
	f.readline()
	labels = []
	for line in f.readlines():
		labels.append(int(line.split(',')[3][0]))
	return labels

def time_features(bid_times):
	if len(bid_times) < 2:
		return None, None, None
	else:
		max_time = bid_times[-1] - bid_times[0]
		min_time = max_time
		avg_time = 0
		for i in range(len(bid_times)-1):
			avg_time += bid_times[i+1] - bid_times[i]
			min_time = min(min_time, bid_times[i+1] - bid_times[i])
		avg_time /= float(len(bid_times)-1)
	return min_time, avg_time, max_time
	
def make_features(id, bid_times, num_bids, categories_bid_in, auction_bids, countries, ips, auction_wins, devices, address_dist):
	#features: total num bids, num categories, max category, num auctions, max bids in an auction, num countries, num ips, wins/auctions, num devices
	total_bids = num_bids[id]
	cat = categories_bid_in[id]
	if cat:
		max_category = max(cat, key=cat.get)
	else:
		max_category = None
	num_categories = len(cat)
	auct = auction_bids[id]
	num_auctions = len(auct)
	if auct:
		max_in_auction = max(auct.values())
	else:
		max_in_auction = 0
	num_countries = len(countries[id])
	num_ips = len(ips[id])
	win_percent = auction_wins[id]/float(num_auctions) if num_auctions > 0 else np.nan
	num_devices = len(devices[id])
	min_time, avg_time, max_time = time_features(bid_times[id])
	features = [total_bids, num_categories, min_time, avg_time, max_time, num_auctions, max_in_auction, num_countries, num_ips, win_percent, num_devices, address_dist[id]]
	#features.extend(category_features(max_category))
	return features
	
	
def num_same(u1, u2):
	num = 0
	for i in range(min(len(u1), len(u2))):
		if u1[i] == u2[i]:
			num += 1
		else:
			break	
	return num
	
def gen_features(user_file, bids_file):
	bidder_ids = set()
	bidder_id_list = []
	address_dist = dict()
	users = open(user_file)
	users.readline()
	for user in users.readlines():
		user = user.split(',')
		bidder_id = user[0]
		bidder_ids.add(bidder_id)
		bidder_id_list.append(bidder_id)
		
		address_dist[bidder_id] = num_same(user[1], user[2])

	bid_times = defaultdict(list)
	num_bids = defaultdict(int)
	categories_bid_in = defaultdict(lambda : defaultdict(int))
	auction_bids = defaultdict(lambda : defaultdict(int))
	countries = defaultdict(set)
	ips = defaultdict(set)
	auction_winner = dict()
	auction_wins = defaultdict(int)
	devices = defaultdict(set)
	
	
	bids = open(bids_file)
	bids.readline()
	for bid in bids.readlines():
		bid = bid.split(',')
		bidder_id = bid[1]
		if bidder_id in bidder_ids:
			auction = bid[2]
			merchandise = bid[3]
			device = bid[4]
			bid_time = bid[5]
			country = bid[6]
			ip = bid[7]
			
			
			num_bids[bidder_id] += 1
			categories_bid_in[bidder_id][merchandise] += 1
			auction_bids[bidder_id][auction] += 1
			countries[bidder_id].add(country)
			ips[bidder_id].add(ip)
			auction_winner[auction] = bidder_id
			devices[bidder_id].add(device)
			bid_times[bidder_id].append(int(bid_time))
			
	for auction in auction_winner:
		auction_wins[auction_winner[auction]] += 1
	
	features = []
	for user in bidder_id_list:
		features.append(make_features(user, bid_times, num_bids, categories_bid_in, auction_bids, countries, ips, auction_wins, devices, address_dist))
		
	return bidder_id_list, np.array(features)
	

def output_to_file(ids, predictions, file):
	file_strings = ['bidder_id,prediction']
	for i in range(len(predictions)):
		a = '0.0' if predictions[i] == 0 else '1.0'
		file_strings.append(ids[i] + ',' + a)
	f = open(file, 'wb')
	f.write('\n'.join(file_strings))

def normalize_features(features):
	imp = preprocessing.Imputer(missing_values="NaN", strategy='mean')
	imp.fit(features)
	features = imp.transform(features)
	return preprocessing.MinMaxScaler().fit_transform(features)

def train_model(features, labels):
	classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=200)
	classifier.fit(features, labels)
	return classifier

def train_and_test(features, labels):
	print len(features)
	n = int(len(labels) * 4/float(5))
	print n
	classifier = train_model(features[:n], labels[:n])
	
	predictions = classifier.predict(features[n:])
	print len(predictions)
	print sum(predictions)
	correct = 0
	total = 0
	true_positive = 0
	for i in range(n,len(labels)):
		if predictions[total] == labels[i]:
			correct += 1
		total += 1
	print correct, total, correct/float(total)

	
bidder_ids, features = gen_features('train.csv', 'bids.csv')
print 'features generated'
labels = get_labels('train.csv')

pickle.dump([bidder_ids, features, labels], open('data.pkl', 'wb'))
test_ids, test_features = gen_features('test.csv', 'bids.csv')
pickle.dump([test_ids, test_features], open('test_data.pkl', 'wb'))
#bidder_ids, features, labels = pickle.load(open('data.pkl'))
#test_ids, test_features = pickle.load(open('test_data.pkl'))

#sample distributions
'''
bidder_ids2 = list(bidder_ids[:100])
features2 = list(features[:100])
labels2 = list(labels[:100])

for i in range(50,len(labels)):
	if labels[i] == 1:
		labels2.append(labels[i])
		features2.append(features[i])
		bidder_ids2.append(bidder_ids[i])

labels = np.array(labels2)
features = np.array(features2)
bidder_ids = np.array(bidder_ids2)
'''
all_features = np.concatenate((features, test_features))

all_features = normalize_features(all_features)
train_features = all_features[:len(features)]
test_features = all_features[len(features):]
print 'training model'
#train_and_test(features, labels)
model = train_model(train_features, labels)
predictions = model.predict(test_features)
print sum(predictions)
output_to_file(test_ids, predictions, 'output.csv')

#ADD MINIMUM TIME BETWEEN THEM AND LAST AUCTION BID
#ADD FIRST BID