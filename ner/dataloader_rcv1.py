from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()

# print the information of the data
print(rcv1.data.shape)
print(rcv1.target.shape)


# split data into test and training portions
training = rcv1.data[:23149,:]
test = rcv1.data[23149:,:]
