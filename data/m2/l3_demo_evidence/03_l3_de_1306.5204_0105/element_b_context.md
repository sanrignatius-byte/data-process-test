# Element: 1306.5204_table_2

- **type**: table
- **doc_id**: 1306.5204
- **label**: Table 2
- **page_idx**: 0

## Caption

Table 2: Average centrality measures for Twitter retweet networks for 28 daily networks. “All” is all 28 days together.

## Content

Table 2: Average centrality measures for Twitter retweet networks for 28 daily networks. “All” is all 28 days together.

## Context Before

Comparison with Random Samples In order to get additional perspective on the accuracy of the topics discovered in the Streaming data, we compare the Streaming data with data sampled randomly from the Firehose, as we did earlier to compare the correlation. First, we compute the average of the Jensen-Shannon scores from the Streaming data in Figure 6, $S$ . We then repeat this process for each of the 100 runs with random data, each run called $x _ { i }$ . Next, we use maximum-likelihood estimatio

Results of this experiment, including $z$ -Scores are shown in Figure 7.

different communities with each other or funnel different information sources. Furthermore, we calculate the Potential Reach which counts the number of nodes that are reachable in the network weighted with the path distance. In our Twitter networks this is equivalent to the inverse in-distance of reachable nodes (Sabidussi 1966). This approach results in a metric that finds sources of information (users) that potentially can reach many other nodes on short path distances. Before calculating these measures, we extract the main component and delete all other nodes (see next sub-section). In general, centrality measures are used to identify important nodes. Therefore, we calculate the number of top 10 and top 100 nodes that can be correctly identified with the Streaming data. Table 2 shows the results for the average of 28 daily networks, the min-max range, as well as the aggregated network including all 28 days.


## Context After

identifying ${ \sim } 5 0 \%$ key-players correctly for a single day is reasonable, and accuracy can be increased by using longer observation periods. Even more, the Potential Reach metrics are quite stable for some days in the aggregated data.

Network-Level Measures

We complement our node-level analysis by comparing various metrics at the network level. These metrics are reported in Table 3 and are calculated as follows. Since retweet networks create a lot of small disconnected components, we focus only on the size of the largest component. The size of the main component and the fact that all smaller components contain less than $1 \%$ of the nodes justify our focus on the main component for this data. Therefore, we reduce the networks to their largest component before we proceed with

From December 14th, 2011 - January 10th, 2012 we collected tweets from the Twitter Firehose matching any of the keywords, geographical bounding boxes, and users in Table 1. During the same time period, we collected tweets from the Streaming API using TweetTracker (Kumar et al. 2011) with exactly the same parameters. During the time we collected 528,592 tweets from the Streaming API and 1,280,344 tweets from the Firehose. The raw counts of tweets we received each day from both sources are shown i

different communities with each other or funnel different information sources. Furthermore, we calculate the Potential Reach which counts the number of nodes that are reachable in the network weight

## Referring Paragraphs

1. Table 2 shows the results for the average of 28 daily networks, the min-max range, as well as the aggregated network including all 28 days.

2. different communities with each other or funnel different information sources. Furthermore, we calculate the Potential Reach which counts the number of nodes that are reachable in the network weighted with the path distance. In our Twitter networks this is equivalent to the inverse in-distance of reachable nodes (Sabidussi 1966). This approach results in a metric that finds sources of information (users) that potentially can reach many other nodes on short path distances. Before calculating thes

3. different communities with each other or funnel different information sources. Furthermore, we calculate the Potential Reach which counts the number of nodes that are reachable in the network weighted with the path distance. In our Twitter networks this is equivalent to the inverse in-distance of reachable nodes (Sabidussi 1966). This approach results in a metric that finds sources of information (users) that potentially can reach many other nodes on short path distances. Before calculating thes

4. Table 2 shows the results for the average of 28 daily networks, the min-max range, as well as the aggregated network including all 28 days.
