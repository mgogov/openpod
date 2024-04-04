**Daniel Whitenack:** Welcome to a very special episode of Practical AI. This is Daniel Whitenack. I am a data scientist with SIL International, and I'm joined as always by my co-host, Chris Benson, who is a strategist at Lockheed Martin. How are you doing, Chris?

**Chris Benson:** Hi. I'm doing very well. I love the way you intro-ed that, about a very special episode, because I am excited about this, too. I'm gonna get out of the way and let you introduce it.

**Daniel Whitenack:** Ha-ha! Well, you know, Chris, we've talked over the years about how there's so much amazing AI work going on all around the world, and that we want to feature more of that work that's going on outside of the U.S, outside of Western Europe... And I'm really excited, because recently I got connected with the Open For Good Alliance. This Open For Good Alliance is a multi-stakeholder group of about 13 members, which was formed in 2020. So that includes the International Development Research Center in Canada, GIZ's FAIR Forward, Artificial Intelligence for All Project, and Makerere University... And we're really privileged today, because we have two individuals from Makerere AI Lab with us, Joyce Nabende and Mutembesa. Welcome.

**Mutembesa:** Happy to be here.

**Daniel Whitenack:** Great to have to you.

**Joyce Nabende:** Yeah, thank you, Daniel. Nice to be here.

**Daniel Whitenack:** We're going to work with the Open For Good Alliance, and in particular, Joyce, you've agreed to join us for some follow-up episodes in a spotlight on AI in Africa podcast series... So we're really excited to have you both join us today to talk about the great work happening with Open for Good, and Makerere University... But also to follow up with you, Joyce, on future episodes featuring some other AI researchers and developers in Africa, and about some of the AI community-building that's going on in Africa. It's really exciting.

Joyce, how originally did you get involved with this Open for Good Alliance and some of the community-building, dataset-building that's going on with that?

**Joyce Nabende:** \[03:59\] Yeah, thanks, Daniel. It's exciting to be joining this episode, and also the future episodes that are coming up. Joining Open for Good - that's a project that we've been working with Mozilla and FAIR Forward around building resources for African languages, particularly Luganda. The project that we're doing there is 1) trying to build our automatic speech recognition systems. But we realized that you can't build a system without data, and so we had to go back and then think of how can we be able to collect the data that we can use for building our systems, and that's how the connection came in with Mozilla and FAIR Forward, because Mozilla has an open platform where we can be able to crowdsource voice recordings.

But before we even get to crowdsourced voice recordings, we needed our text recordings, so that brought in the need for data that is localized, that is built upwards; moving down, moving upwards. So because of what was going on with that project in Uganda, at the Makerere AI Lab, but also within Africa, we thought that our Open For Good Alliance is a good alliance; so when we formed this alliance, what we acknowledged mainly was that there was a lack of localized training data that was of sufficient quality, and that was one of the major obstacles for local AI innovation in Africa, but also in Asia. So because of that, the alliance was formulated to provide a platform for the coordination and exchange of good practices on how to increase the availability and the quality of openly available training data for machine learning.

And although I just gave an example of NLP data, or speech data, there are several kinds of data that we are seeing the need for to develop and to work on in the African context. So the alliance is an association that brings organizations that are working on different kinds of data.

**Daniel Whitenack:** Yeah, that's so awesome. I know that Chris as a strategist with a major organization is always telling me data strategy is a key piece to the overall AI strategy within any organization... So it's awesome to see this alliance form to really work on those localized datasets.

I know that Makerere University is also one of the founding members of this alliance... Mutembesa, you work as a research scientist within the Makerere AI Lab... Could you tell us a little bit more about that lab, and some of the things that you're involved with, and the membership in that lab?

**Mutembesa:** Thank you for having me over, Mutembesa here. So at the Artificial Intelligence Research Lab here in Makerere University this is an effort that -- it's pretty much a group actually, once you get to look a bit closely, because there's members that have been through the pipeline, through the system, at the Makerere University here in Kampala; it started as a group of people doing their doctorates in the 2009-2010, about that period, and they were coming back from exchange programs that were with other universities abroad... So they returned to be able to use these computation techniques that they've learned to solve issues that were pertinent to the local community, largely Africa.

Some of the work that is majorly in there - there's a lot of work in agriculture, because the focus had to be on issues that were of interest to the people, or of interest to the communities... So agriculture, health, looking at infrastructure, languages now... So some of these - for each maybe I'll just highlight some of the works that have been done in there.

For agriculture, the lab or the group has had a strong contribution to data representation of, say, crop diseases and pests on a large scale, being able to crowdsource that from communities or farmers with mobile phones.

There's also been work on automating mundane tasks that are being carried out by experts... So sort of using a lot of machine learning and AI to be able to do, say, disease recognition and identification, and classifying those diseases.

\[08:12\] There's been some work that has been around being able to diagnose the plants non-invasively, using spectrometry light to be able to identify or classify the kinds of diseases that it matches with for some of the key crops. And when I say crops, the early efforts of the lab have been very focused on food security crops. Since about 2010, all the way down to about 2018, the lab has been focused on food security crops, but now we see a greater divergence to other important income developing crops, or crops that improve the nutrition... There's been work on being able to use radio, because radio is still the biggest social media here in the global South, especially here in Africa... So to be able to use radio, which Joyce was lead. To be able to use radio, to map where crises are for different crops, or for different diseases, or for different pest infestations, or whatever topics that are around diseases.

And that's just the highlight of the work that has been done. This is in monitoring and evaluation for crops. We've also had work around being able to use AI to make accessible credit scoring for historically unbanked small holder farmers, amongst many other things. So this is just the tip of the iceberg of some of the works that have been done in agriculture.

When we move over to health, some of the prominent work that we see has been around being able to produce artifacts that can then be attached to microscopes in healthcare centers. Why is this important? It's because largely we have a ratio of 100-200 patients for every clinician or every lab tech... Whereas the gold standard is like 1-20. So you find that there is a need for being able to reduce the load on clinicians or on lab technicians.

Also, beyond that, once you are able to get that data, being able to use machine learning to identify which parasites on some of these microscopes you're looking at, and then to be able to do a count. This would reduce the load and the 30 minutes procedures to about a two to five minute procedure. So it means that clinicians or the lab techs can work on more people effectively throughout the day within a reduced cost. We've done some work where we're able to use machine learning to identify and do the counts within an ethical and responsible kind of way.

Also in health we have work that was previously done around using mobile/cell phone tower data to track the mobility of people. This data was from a prominent telecom; again, ethically anonymized to be able to just provide a network of how people move, and then be able to use that as a feature for predicting the spatio-temporal patterns of diseases where the contributor is -- you know, somebody gets infected here, travels to another place, gets beaten by another vector. So some of the diseases like malaria, where mobility is a contributing factor.

This is just an overview of some of the work in health. There's definitely much more in infrastructure... Some of the early work that has come out of the lab has been being able to identify motorcycles, trucks within traffic using very local \[unintelligible 00:12:03.17\] sized devices to be able to identify and know "Okay, this road is probably jammed" and predict where traffic scenarios are going to be. Why is this important? It's because there's very limited resources around city management or township management here.

\[12:23\] So those are some of the early works. Of course, recently there's been work that is using machine learning on Covid response, Covid data and response. That was started at the height of the pandemic last year.

There's also been work around being able to connect farmers to markets using their small batten phones - not too sure if you know them - where a farmer and a willing buyer can send their requests to a central place and there a machine learning matching algorithm could be able to match who is the most potential buyer, and the most potential seller, based on the proximity of the price, geographical distance, to multiple other features. Of course, this gets better and better as you have more data coming in.

Also, maybe just one of the last that I would like to highlight is we have a project that is looking at the ethics, the fairness, accountability and transparency of some of these algorithms that we build... Because our policy or our mandate is so paper thin that even doing basic research within the global South you end up impacting lives of people. So our permeation of work is very paper-thin that we always end up working with communities directly... Which is one of the three ethos of the lab that I will talk about just after this.

So one of the things that we also have to look at is what are the ethical implications of working within these communities, sort of measuring our impact and what are the kinds of fairness questions that we have to ask.

Wrapping that up -- there's a couple of other projects, but wrapping that up, this is based in a three-step ethos for the lab, where the first is to be able to find a good local problem; that is the first ethos that we follow. That means a problem that matters, a problem that has democratic voice as being important. Then secondly, being able to match that problem to a good computational toolkit... Or once we have a problem, we try from a research point of view to see, "Does this match some technological or computational solution that is accessible to us?" Within AI, or within machine learning, or within the computing.

Then the last is to be able to tie the challenge, the technique to a local beneficiary. So pretty much every project that you will hear out of the lab, every once single project has a local community attached to it, or has some beneficiaries. If it's health, there is a hospital that we are attached to. If it's languages, there's local radio stations that we're attached to. If it's agriculture, we're attached to the national crop service, we're attached to local farmer communities. If it's in roads, we are attached to the city management. If it's air quality monitoring, which is one of the works that has also been done at the lab by a gentleman called professor engineer it's also attached to city management, to schools who have vulnerable communities.

**Daniel Whitenack:** So Joyce, Mutembesa gave an amazing intro to all of the things happening at Makerere AI Lab, which are just spectacular in terms of all the different projects that you're involved with, and the amazing work that you're doing... He talked about this ethos of working on a problem that matters, and connecting a data and computational toolkit to that, and also sort of attaching that problem to a beneficiary.

\[16:07\] I know as someone working in a non-profit, but also having talked to a lot of people about AI for good, or social impact projects, one of the things that can happen is that AI people can develop really great and interesting technology, but maybe that technology doesn't always benefit the end user or the local community that they might have in mind. How do you think about that as a lab and make sure that the problems that you're choosing and how you're going about those solutions end up impacting these local communities in a beneficial way?

**Joyce Nabende:** Yeah, thanks, Daniel. I think that's a very good question, and a very pertinent question, especially when you're developing AI for social good. I guess - how do we try...? It's a larger process, so how do we try to ensure that we do this? Some of the issues come up organically. For some, the people who actually need the AI tools kind of approach the lab and say "We want to work on a project with you." And that comes in, for example, with the project that we are working on for building tools for breast and prostate cancer diagnosis, to an MRI and ultrasound. The people who actually need the technologies were very fascinated by AI, and then they learned about the AI lab. Then they came to us and then we've begun to work together.

First, of course, we started by writing joint proposals together, but eventually, we end up building the tools together with them. Many times when we do that, we actually have meetings with them, they host us at the center where they actually do the testing and the actual recruitment of the patients, and then they take us through the process. Also, as we build the models concurrently -- we do this concurrently with them, and then we are able to get feedback from them, because they are very instrumental. Sometimes you look at an MRI and you don't know where the cancer is, or the lesion is, and they come in very handy and say "Okay, this is where the lesion is. This is how you're going to label this image." And eventually, when you build that tool and you go to test the tool, then this is something that is usable for them, because they've been part of the process, or part of the journey of building the AI model. So that's the thing that we've been working with in health.

Then in agriculture somehow also it's been like that. Before we even have a project assigned. So if we get maybe funding, and we are beginning a new funding phase or a new project in the lab, we actually go to them. We have the experts that we work with, the agriculture experts. We go to them and we discuss the idea that we have, and we want to implement, and the project that we want to implement... And it's only after they've understood what we want to do that we get a sign-off from them. Then we get the support that we require to start building that technology.

So we are very intentional when we build the AI tools that we don't build them just to have a good tool or a good model, but this model that should actually be able to work and perform in the field for which we are developing the model. So we work very closely with the agriculture experts... For example, we work very closely with the small holder farmers who are going to use the tools. Many times we have farmer training where we have a tool that we have developed, for example a tool to give them recommendations, and we try as much as possible to hold workshops, and trainings with them. During Covid it's a little bit difficult, but we aim to have physical workshops where we take them through the technology, as well as bring in the agriculture experts, because sometimes they might ask a question that's not really necessarily related to the technology, but it's related to the domain, and that's where the agriculture experts can come in, during those workshops, during those trainings.

Then we also get feedback from them, from the farmers who are going to eventually use the technology on what they think or what they like or what they don't like with that technology. And if we can't have the physical trainings, also through the lab we have a dedicated call center where we have the people who are calling and checking on the farmers to find out if they have any problems with that technology, any problems with the tools that they are using. Then we get back this feedback, go through the feedback, and try and improve the models that we are building.

\[20:12\] So it's both that -- sometimes it's \[unintelligible 00:20:13.13\] but also other times it's the experts who come to us, and then we are able to build the technologies that are very impactful, and we are hoping that these technologies also can be usable... Because we don't want at the end of a funding phase that the technology ends there. We want something that is continued in usage of the AI tools that we are building.

**Chris Benson:** Joyce, I love the way you approach the problem - it's just delightful to listen to - with the emphasis on trying to find solutions to problems that are specific to the African community at times, and those focuses and these creative things like having call centers to reach the farmers and others that you are working with. How did you come up with this particular model to serve this community that you're doing? I'm curious, as you look at the broader world of AI around the globe, what are some of the things that you feel are unique to what your lab is doing, or very differentiated compared to others in the way that you're satisfying the problems that you're addressing at this point?

**Joyce Nabende:** I think for the call center Daniel might have more ideas about it... But what happened and how the call center evolved is through our crowdsourcing project what we wanted is we wanted the farmers to be able to send us images of their gardens, so that we can be able to build models that can map what is taking place in that garden over time. But because the farmers are out there and we're introducing a new technology to them, which is the crowdsourcing tool, we thought it would be interesting that we don't just throw that technology at them and then assume that everything will work out okay, because not everyone is tech-savvy, and these are applications that are not on features phones, these are applications that are on smartphones, and not everyone knew or had an idea of how the smartphone is able to work.

So we felt that there was a need to actually reach out and follow up on the crowdsourcing... Because also -- sometimes you give them technology and then maybe you don't hear from a farmer maybe after a week, and you're wondering, "Oh, what happened? Is everything going okay?" because you're running a project. So we thought it was intentional and we thought it was good to try and reach out to the farmers through the call center. We think that this has helped us to gain traction in terms of output, and not only on the crowdsourcing, which we thought was beneficial for us, because we were getting the image data, but in terms of the farmers continually continuing to use that technology that we have developed.

So what we think is unique is the fact that when we are developing these tools that 1) we are involved together in the data collection - I think that data is always the driving factor, so we are involved with the people from which we are collecting data. So if for example we are looking at health as an example, we go and we are very involved in the collecting of that data, we are introduced to the place where that data will come from... So we're not like at the end of the computer, waiting for the data to come from oh let's build a \[unintelligible 00:23:04.17\] and take them back. No, no, no. So it's that you understand, you go for the training with the radiologist and they tell you "Okay, this is how we're going to capture the image, this is how the image looks like", if it's for the land, they'll tell you "This is how the beelines look like..." So there's that involvement that we think is very unique for us.

And also speaking maybe about one of the other projects, which is the air quality project. So with the air quality project what's also unique about that is that, again, with that data collection, that we build the data collection devices ourselves, and then we deploy them together with the community, and collect the data. So I feel like the unique aspect here is that we are involved in that data collection, in that data curation, and then also that along the journey of building the models, that we do this concurrently with the eventual users of the technology... And that's always a good sign, to have acceptability with the stakeholders, with the policymakers... And we always have to make sure that when we have a project, we disseminate. We kind of provide the findings out there; we are intentional to who we invite to the dissemination seminars, because we want to ensure that they get to know the work that's going on in the lab for continuity, but also for scalability to other sectors that can be interested in the AI models and the technologies that we are building.

**Daniel Whitenack:** \[24:21\] I think a lot of the points that you've made there, Joyce, are just so valuable for any data scientist or AI researchers out there that are working on curating and crowdsourcing and analyzing their data - having that connection with the group that's generating the data, and being deeply involved there I think is such an important point. And maybe that gets to some of the ethics and bias type of issues that, Mutembesa, you mentioned. As you're working on crowdsourcing and curating a lot of this data, Mutembesa, maybe you can comment on how you think about bias in those datasets, and work to make sure that your data collection and usage is ethical, and you're monitoring for bias.

**Mutembesa:** Yes, thank you very much. One of the things that we really do is -- so there's two ways to think about it. One of the ways is that you have a regulator or an authority on that kind of data, that is working with you. If it's a hospital and you're looking at working with patient data or clinicians, then you want to also be working with the lab techs. In agriculture specifically, which is sort of where our way in has been a bit greater, is once you're working with groups of farmers, you're also working with a national crop service or national animal service. In that way, what you do is that you end up having a lot of work go into how fair can we be in our data collection geospatially, what are the important areas to go to, so that there is data representation that has equity.

If you're involving multiple teams or multiple groups within a certain community-sensing exercise, you have to have a selection criteria, which is not developed by us alone. That has to be in collaboration with a national regulator, which means that such a selection protocol or a selection criteria is sensitive to different national or community needs... So like, okay, we need people to participate, but there has to be some gender equity in the participation, where we always try to balance participation of both women and men. Of course, there's some degree of sexism, and in some places it's actually a challenge due to some of the cultural norms that you have to overcome.

But we look at the idea of ethics from that fast point - the ethics, the bias and the data, looking at who is collecting that data, geographically where are they going, and being able to involve the subject matter experts, and then provide the subject matter experts... They weigh in, and then we provide the technology that sort of sustainably has to be able to collect this data over time. So that's the first way we look at ethics, or fairness, or transparency, or accountability for these data collection mechanisms.

Coming around all the way at the end of the projects, until recently, we started doing what are known as evaluations where we try and evaluate the impact that we have had on those communities. So we go back to the community -- of course, we have close-knit relationships with them, and so we have regular sessions where we have interfaces with them... But at the end of a certain trial period we go back to them and then we do some impact assessment.

\[28:01\] If we have collected data from you, has the data come back to help you? Ever since we gave you this tool, has this tool been of help to you? And using that kind of technique you can be able to quickly evaluate which parts of the livelihoods of our technologies or this data collection impacted. You know, did it improve some metrics that we thought it would improve on the holistic, end-to-end picture? Has this had a significant impact? Being able to measure that impact with some relative accuracy from a qualitative and quantitative point of view.

So recently, maybe just as an example really, to just be very clear - we had a livelihood assessment; we wanted to know, from 2016 (maybe 2018 more) we have been working with groups of about 200 farmers; we wanted to evaluate how well we have contributed to their livelihoods. Because when we first met these farmers, they were small holder farmers, with a very low income, just enough for their food and for their households... So we tried to evaluate whether our tools that we have put into their lives - that is including a tool to diagnose their plants, a tool for data collection, which they send to the national service, and the national service gets back to them (the national crop service).

We have had a tool which we like to joke and call it like a farmers' WhatsApp, but it's really a question and answer tool that allows them to connect with experts, and also connect with fellow farmers to increase the local knowledge transfer between and among them.

So we evaluated for a period of about three years that we've been trying with about 200+ farmers how has that impacted them financially, socially, economically, intellectually, among other things. We're about to publish this work, and you could definitely see a positive correlation with the use of these technologies, and many of the people that have come out to be leaders. Now many of these farmers - about half or more - are known or considered as village information points, where people within their villages or localities are able to run to them.

So once you start seeing that positive impact trend within the communities, that is, you know gender, then you start seeing the connection between the endpoints and the selection criteria that we had, that sort of tries to minimize bias within the data, within the groups participating, and many other things. That's how we look at bias. It's a two-ended stake for us - before, and then after.

**Break:** \[30:46\]

**Chris Benson:** Joyce, your lab is involved in so many different AI tasks, everything from computer vision and natural language processing, you do work in health... There's a lot of it. As I'm looking through that, I struggle to keep up with all of those topics myself as I follow the field... How do you manage a group that is involved in such a diverse set of tasks, and kind of keep it all together, and in your head, and moving forward productively? \[laughter\]

**Joyce Nabende:** \[32:14\] That's a good question, because I keep thinking, I'm like, "Okay, NLP is really good, and there's a need for that, so we really need to do a project in NLP." And then we get funding, and off we go with the project in NLP. But there's also computer vision, you know? It's quite diverse, but I don't try to do this by myself. Daniel is here, but there are also other colleagues that I work with, both in the college of computing, but also in the college of engineering, design, and technology who also have backgrounds in machine learning and AI, and we co-lead some of the work together. So there's a machine learning lab in that college of engineering that we work together with.

**Daniel Whitenack:** Yeah. And could you describe some of the -- I mean, you already mentioned some of the things you're doing with the Open for Good Alliance, but could you describe... It sounds like there's a number of collaborations that you're fostering even outside of the university as well.

**Joyce Nabende:** Yes. So the way we've designed it is the first service are really, first of all trying to understand, when people want data, where can they get data from? So the first service that we are having is really around the repositories of the data that are available. For example, Radiant Earth, where it's an open repository for satellite imagery, data, and Radiant Earth is also one of the organizations that we have in the Open for Good. So part of the work that we do inside there is to try and publicize and make people know. Because one of the things that people say to us, "Okay, if I want to start AI and I want to maybe do a problem with satellite imagery data, where do I get the data from?" So part of the thing we are doing is awareness, where we are making sure that people are aware about the public repositories that are available there.

But on the other hand as well we have the community building that the organization that we shall talk to eventually, which is Masakhane - this is more specialized in one field, and that's NLP, where they do a lot of community building within the field of natural language processing, and talking to people, and dealing with people... So I think strengthening such organizations can be something that we can be able to work with and talk to.

Then also another organization is -- much of the things that people also need is skilling. So some of the things that we are also doing out of the alliance, not necessarily within, but out of the lab is, of course, Data Science Africa. I think you've heard about Data Science Africa, that does training and skilling, where we have a one week long workshop, and a summer school where people are coming, and training people in AI and data science skills. And because Data Science is Africa-wide, and it comes I think twice a year -- it used to be once a year, but now it's twice a year, and it keeps rotating around different countries... So we thought that in Uganda we would start something that's local here... So we've also began a Data Science Africa local chapter where we can be able to build capacity of AI and data science, but within the different universities in Uganda.

But our focus is not necessarily just on universities, we want to focus on the intersection between the university, the private sector, the government as well, because we feel that the connection between those different bodies is important. So there's work which is not necessarily the project, but there's work that is really focused on capacity building, because we want to grow the next group of data scientists in Uganda, and in Africa as well.

**Daniel Whitenack:** Joyce, it's cool to hear about that community-building, and I'm just amazed -- I do think you probably have super-powers, because you're managing and leading this lab with so many different important projects going on, but also involved in data crowdsourcing and all of that... And Mutembesa, when I'm hearing you talk, I'm actually picking up on another thread which I'm super-impressed with and curious about, which is the fact that you're as an AI lab not only involved in doing AI research at the cutting edge, but you're also involved in actually producing software and applications that people can actually get their hands on and use.

\[36:06\] In industry, like in most AI groups in industry, this is a struggle, to sort of push from development and research in AI into actual application of that AI in production, and in actual software. How have you navigated that within your own work, in terms of taking AI models from that research stage in the lab into maybe a mobile application, or a website, or whatever it is, or offline processing in production, that people actually get their hands on and impact end users? What have been some of the challenges and successes along that way?

**Mutembesa:** Very interesting question. Thanks, Daniel. So one of the things you may want to keep in mind is that though the technical processes of being able to deliver or implement a piece of technology or innovation. Though that is standard, of like, say, software development, the way you work with communities to be able to deliver that is different across the world... So for us it's been a very different journey from the get-go. We did not realize especially how much appetite there was for tools that would assist or ease the burden on, say, farmers, or on clinicians... But you quickly realize as you start to do work within the global South, which is one of the reasons why there's been a big movement from the lab, co-joined with another entity called Data Science Africa, to be able to share those stories, the experiences and resources across the African continent. Because when you look at our path, we sort of skipped many of the computer age, and went directly to a mobile age, or the mobile era.

So the way technology permeates within the African continent is also very different... So this is one of the things that has ended up. Whichever project within the lab started as just a very simple, basic research idea, either somebody doing their master's, or doing their Ph.D, ended up being highly needed with the community. And once you work with a small community of a small cohort of farmers or a small cohort of clinicians, or a small cohort of, say, radio teams, immediately there's an insatiable appetite from the community, because they have been longing for tools like these.

So there is a thin line -- the lines are very blurred on how we've been developing with some of these. It is really hands-on-deck; you're working with tools that are reaching directly to the communities, and of course, that had been earlier permeated within some of the principles for the lab, or for the group, actually. And I think also the group draws a lot of that from the university.

\[39:02\] The university - and I'm sure Joyce can talk about this more - from a holistic, macro level, it looks at research, education, but also outreach. A very, very strong component of Makerere University, that you have to have an outreach arm of your research, as much as it ties into Academia, being an Academia hub.

So there is already a need... So once you're working on any technology, no matter how small, you will always end up impacting people. There is a very thin membrane between the work that we do for Academia, or for school, and the people that it impacts. So it's a very different ball game in terms of the setting. I hope I've tackled that.

**Daniel Whitenack:** Yeah, thank you so much for that answer, Mutembesa. I really appreciate your perspective there, and also emphasizing that no matter what technology you were building, it is going to have an impact on people, and we should keep that in the forefront of our minds.

Well, I'm really just thrilled to have kicked off this AI in Africa Spotlight series on the podcast with this great conversation with Joyce and Mutembesa from the Makerere AI Lab. Joyce, I'm wondering if you might close us out by just giving us an idea about what you're excited to talk about and discuss as we have some follow-up episodes about other things that are happening in different areas and in different ways as related to AI in Africa.

**Joyce Nabende:** Yeah, thanks, Daniel. So for me it's exciting, the series that we are going to have, 1) to really understand especially the community building, the groups that are out there in Africa that are building communities, that are training communities, that are providing support for communities. I think that's going to be very interesting for us to listen to.

Another interesting thing is to really look at the big problem that we have around the data collection, data curation, making sure that we have data that's not biased... I think you hinted a little bit about it in the questions that you asked around how do we deal with biases... So for me it's interesting to hear how other people have been able to deal with that, to make sure that the data being collected doesn't have bias, it's representative, is inclusive... That's interesting for me to hear about.

The feminist AI is also something that's interesting as well, and I'm hoping that we can be able to hear more about it in the coming series. Of course, we're still in the Covid-19 pandemic. So also hearing about the work that's being done in the African context around dealing with Covid and an integration between using AI for fighting the Covid-19 pandemic - I think that's also something that's interesting, that I am looking forward to hearing. So yeah, I'm very excited for the next episodes that we are going to have.

**Daniel Whitenack:** Yeah, thank you so much, Joyce. Thank you for agreeing to join us on this journey, and thank you, Mutembesa for joining us in this kick-off episode. I appreciate both of you taking time, and looking forward to having the follow-up conversation soon. We'll talk to you all soon. Bye!

**Joyce Nabende:** Bye.