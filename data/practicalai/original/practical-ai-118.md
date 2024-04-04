**Daniel Whitenack:** Welcome to another episode of Practical AI. This is Daniel Whitenack. I am a data scientist with SIL International, and I'm joined as always by my co-host, Chris Benson, who is a principal emerging technologies strategist at Lockheed Martin. How are you doing, Chris?

**Chris Benson:** I am doing very well. We're just in the normal late December holiday rush as we record this... All is well.

**Daniel Whitenack:** Yes, trying to get all those year-end projects that you promised were gonna be done by the end of 2020 done?

**Chris Benson:** I've already failed on that, because I'm already on my holiday break now, as of today. It's my first day.

**Daniel Whitenack:** Oh, congrats.

**Chris Benson:** That's a done deal. Anything I didn't do is into 2021, so I can walk back in with excuses next year.

**Daniel Whitenack:** Yeah. Not that there would be any reason this year -- you know, nothing happened this year that would throw projects off, or...

**Chris Benson:** I don't know what. This whole pandemic thing, you name it - it's been quite a crazy year. I'm optimistic, with vaccines out, that 2021 will be a much better year for all.

**Daniel Whitenack:** Yeah. Well, I'm looking forward to our Christmas at home, with a small group, some good food, but also just sort of a break and a nice relaxing time.

**Chris Benson:** Reset.

**Daniel Whitenack:** Yup, reset and then jump back into the year. I hope all of our listeners have a wonderful break as well. If you happen to be working on fun side-projects during your break, let us know in our Slack channel, or on LinkedIn, or Twitter, or somewhere. We'd love to hear about those.

I'm really excited, Chris... One of the things that I know that we've been requested a couple of times in our Slack channel is a little bit deeper dive into financial applications of AI, from someone that has that expertise... And we definitely have that someone with us this week. This week we have Madhurima Khandelwal, who is vice-president and head of American Express AI Labs. Welcome, Mads. How are you doing?

**Madhurima Khandelwal:** Hello! I'm doing well. How are you all?

**Daniel Whitenack:** Doing great. It's wonderful to have you here; we appreciate you taking the time. Before we jump into all things AI and finance, could you just give us a little bit of a picture of your background and how you got interested in AI and ended up doing what you're doing now?

**Madhurima Khandelwal:** Sure. Personally, I am born and brought up in New Delhi, India, and I have spent over 15 years in American Express. In this tenure, I have held a variety of roles, with increasing responsibilities... I'm currently the vice-president and the head of AmEx AI Labs. In this role, I lead the charter of building state of the art AI products and capabilities which solve high impact, complex business problems across the company needs.

My teams are based out of Bangalore and New York, and if I reflect back, in this tenure I have always taken roles where I feel I've been in unchartered territories... But at the same time, I've always felt supported in every step of my career to pursue these opportunities as well. So that's a little bit about myself.

**Daniel Whitenack:** Awesome. That's so cool. I'm wondering -- so you mentioned this American Express AI Labs... I know that it's fairly common in especially larger organizations to have an innovations lab, or even set up a specific lab for AI applications, because it's sort of a newer thing that maybe the rest of the organization doesn't have a lot of experience with, or it needs special prototyping... Is that the kind of model at American Express, or what does Labs mean? Is it more of a research thing, or how does that work?

**Madhurima Khandelwal:** I think on the contrary, American Express has been investing in the space of AI/ML for a number of years now... And the way the Labs has shaped up as of today - it has a role to play when it comes to research, but it also has a role to play where we provide tools and platforms to our modeling community, so that they're able to utilize AI/ML in a manner that can drive business impact.

\[08:24\] We also have multiple AI/ML teams sitting within our business units which are really excelling in their own domain. But Labs really comes in to solve for those horizontal, company-wide needs that may exist, and may need to get into a specialization zone which these specific teams may not have the expertise in. But I wouldn't say that AI/ML sits really only within this Lab; it's pretty much integrated across the company domains - be it risk, be it marketing, as well as servicing.

**Chris Benson:** I am always curious, I like to ask people - as you moved into taking charge of this capability within American Express, and if you look at just like not the whole career, but maybe just the last couple of positions, were you like the natural person, or was this something where you said "This is a really cool thing. I wanna lead, and I'm volunteering"? How did you get into it? I love hearing the stories about how people got into this space, because they always different. So what was yours, in terms of the short-term?

**Madhurima Khandelwal:** So I know you're asking me for the short-term, but I'll take you back to 2005, when I joined...

**Chris Benson:** Okay, that's fine. Whatever you wanna do, it's good.

**Madhurima Khandelwal:** ...because at that time my role was pretty much around marketing and customer acquisition. We were building traditional models to be able to solve that problem. But as my role matured, we got into the space of digital marketing, which really led to a plethora of volume of offers coming into the ecosystem, and the only way we could solve that problem and yet be relevant to our customers is really venture into the zone of machine learning. That's where this entire interest came into being, where we started using AI and ML so that we could personalize our digital assets for our customers. When you ask me whether I was the natural choice, I think it was a two-way street. This was an area that was of huge interest for me, and thankfully, my leaders also saw my ability to be able to create these solutions, but also create it in a manner that could drive scale... And hence I am where I am.

**Chris Benson:** Gotcha. That sounds great. You were the natural person who could do it, because you were the person who could make it actually happen.

**Madhurima Khandelwal:** Well, I'll take that.

**Chris Benson:** Okay, fair enough.

**Daniel Whitenack:** Yeah, it's always interesting to me too that this sort of space - there is a very close connection to research that's going on, whether that's academic research and that's coming into a company, or whether that's actually research within the company itself... How has it been for your AI Labs in terms of actually productizing research, and the balance between researching something that may never be able to be productives, versus specifically scoping out things that have a product in mind?

**Madhurima Khandelwal:** Yeah, I think often research doesn't really start with the mindset that we will be able to productize it. I think that there's a huge learning and even failing at a research, because you learn something even then. And while often there would be a business problem at hand that you're trying to solve, but AI Labs is a team of Ph.D's and \[unintelligible 00:11:45.24\] who are always challenging the status quo. And when you challenge the status quo, you pretty much go about researching what may be industry-best. What may be something that we could utilize not immediately, but maybe three years down the line.

\[12:03\] In that process, there is learning for us where we may not be able to productize the entire solution, but we may be able to carve out pieces from it which could be relevant to what we are solving today, or even in the future. I think there are stories of enough successes, but there are also stories of failures, and I as a leader am quite proud of those failures as well. That's what has made us learn what we need to do differently as we venture into these researches again.

**Chris Benson:** One of the things you talked about was making that evaluation for what three years out might be... How do you evaluate that, given that this field is moving fast and you have the business drivers that are pushing you in the directions that you need things to go for your business? That's kind of a combination of the technical forecasting and the business forecasting... How do you and your team make those kind of judgments, given what is clearly not enough information at any given point in time?

**Madhurima Khandelwal:** I think it's a combination of one knowing the business well, but also one being strong technically. I'm glad we are a team which actually have a great combination of the two. Often, we're able to solve problems not because we're asking a team "What is your AI/ML problem?", the question that we often ask is "Tell us what is the day-to-day process that you do today?" And when somebody explains to us what their process is or what their product is, often ideas are generated together as to how we could make that better using AI/ML.

Let me take a few examples for you. What I was talking about earlier on in the career, when we were building traditional models, and personalization in the space of digital assets, be it the website, mobile app, email - the problem at hand was "We have this shelf full of offers. What we don't know is which one to pick out when our customer is on the channel." And the only way we could solve that was to build best-in-class machine learning algorithms.

Another example -- so this example that I talked about was for our external customers, but another example here is for our internal customers or colleagues, where we have started to integrate AI in operational functions as well. And that's really speeding up our manual processes. If you were to ask these teams, they would spend hours in doing these processes themselves, but our products are now able to free up that time for them, so that they can spend time into doing deeper analytics and more complex tasks. One example of that is how we have created a tool for our vendor management team, that pretty much identifies potential duplicate invoices, which otherwise they were spending so many hours figuring it out themselves.

So I think these are areas where we are driving solutions not because we ask the question of "What is the AI/ML need that you have?", but because we understood the problem at hand.

**Daniel Whitenack:** Yeah, something interesting to me as you're speaking about all of these solutions, that some of them are maybe not what first comes to mind when people think about AI in finance... So maybe some people think of like "Oh, quants on Wall Street, that are optimizing trades", or something like that. Maybe other people think of fraud detection, or risk analysis, which is something that comes up a lot, and I'm sure is very relevant still within American Express - we'll hopefully talk about soon - but it seems like there's these other applications too that are really coming out of the fact that American Express is a large organization; you're dealing with a lot of, let's say, documents... Or you also have customer support type issues, or marketing type things that you have to deal with...

\[16:04\] As I'm looking through the website of the AI Labs a little bit, it talks about natural language processing, document recognition and processing - those two things, like NLP and these sort of computer vision things might not be the first things that come to my mind when I'm thinking about AI in the sort of financial vertical. What is the balance on your team? Is there a lot of that operational support that you're doing internally, or is a lot of the focus on direct models that impact your actual financial products? What's the balance there, and how do you think about that?

**Madhurima Khandelwal:** So I wouldn't say the focus is primarily on the automation part or the document processing part. I think the focus of the use of AI/ML in the company is really within credit risk, marketing and servicing. That's really what is impacting our external customers. That's where the focus has been, and we have obviously created or improved our overall usage of machine learning in just advancing in this space.

Yes, while we have been mastering that art, we have also started to invest into areas of NLP, of automation, and driving value in that as well. But as you said, Daniel, I think the story is left untold if we don't talk about fraud prevention, which is where it all started within American Express. And for the listeners, fraud prevention is really one of the first areas where we deployed machine learning models. This was back in 2010. We saw a dramatic increase in our ability to detect fraud with the usage of these machine learning models.

It goes back to what we believe in, that we would like to have our customers' backs, and really servicing our customers is our top priority, and keeping fraud rates low is key to achieving this goal.

**Break**: \[18:20\]

**Daniel Whitenack:** Mads, I love it that you started getting into this fraud prevention topic, because I know one of the things when I'm teaching AI workshops, a lot of the examples that I might give when I'm first introducing machine learning or AI, or maybe in courses online or something like that, people talk about fraud detection, and they have some dataset where it's maybe a set of numbers indicative of transactions, or a person's history, and then a 1 or a 0 for fraud or not fraud, right? And I assume -- and I'm very curious about this personally... I assume that the situation is definitely much more complicated than that. There's different types of fraud, there's different data that's relevant to those different types of fraud... Would you be able to just give us a little overview or a sense of what types of fraud are your primary concerns, and what data is related to that fraud?

**Madhurima Khandelwal:** \[20:23\] Yeah, absolutely. I think when it comes to fraud, the number one problem that we are faced with, even in the space of online fraud, is really our ability to detect fraud in real time basis. What that means is that when a transaction is actually taking place, can we make a decision whether or not that transaction is fraudulent.

If you think about American Express transactions, we have more than $1.2 trillion transactions annually, and we would like to create this decision in almost milliseconds, when the transaction is taking place. And that is where the need for really building machine learning models that can be deployed in real-time becomes the number one ask, so that we're able to service our customers.

So while the fraud definition or the data has remained 1/0, it has magnified basis the number of transactions that exist globally, but at the same time with digital coming into play, you have online transactions happening; it becomes difficult for you to detect fraud, because it's often not happening at the physical address of the resident... So a variety of, I would say, identifiers which feed into our models to be able to detect that.

But what is more critical is our ability to really run this algorithm in real-time, which is where a lot of the engineering, a lot of the architecture is coming into play to deploy these models.

**Daniel Whitenack:** That number you said definitely caught me. It was something over a trillion transactions a year, or something like that, which I think if my rough/quick math comes out, that's like more than a couple million a minute, which is crazy to me.

**Chris Benson:** That's called scale.

**Daniel Whitenack:** Yeah... \[laughs\]

**Madhurima Khandelwal:** That's called scale. Yes, that's right.

**Daniel Whitenack:** Yeah, so I can definitely confirm that I have never done two million inferences in a minute with any of my models... \[laughs\]

**Chris Benson:** Let me ask a follow-up... And it might not be a fair question, because I'm not sure if this is specific... But I'm curious - are there certain types of fraud, in the sense of like, is it just fraud and there's one generic type where something comes in, or are there different classifications of fraud that you're looking for? I was mainly asking that as a follow-up. And then as you address that, I'm kind of curious, what do you do with it? Because you're talking about real-time there, and you may have a customer on the phone, or transactions are coming in... How do you integrate the output of the model, whatever that fraud is, into your actual operation, so that it's usable in that real-time, or maybe near-real-time context that you're having to deal with?

**Madhurima Khandelwal:** I think I may not be able to provide a lot of details about fraud modeling...

**Chris Benson:** No worries. And I wasn't sure if that was a fair question to ask anyway, so if you want, you can move right on from that... Because in some cases, when we talk with folks we don't know what their area specifically is and what it isn't. So go straight to the model question - you put it through, you mentioned that real-time capability... What happens at that point? Not just with the model itself, but the output of the model - how does that get used in the real world? What does that look like for companies who don't have that capability at this point? What does that integration of model output and humans dealing with stuff in real-time look like?

**Madhurima Khandelwal:** \[24:10\] Sure. I think I'll go back to the example of personalization on our digital assets, where the data scientists are really building the best-in-class models. So they are pretty much figuring out for a given problem where you want to surface up relevant content for your customers, how would you really go about figuring out what is relevant at that point in time when the customer is on the channel?

But while we build this model, you want to be able to pretty much run this model in real-time, so that you're able to surface up the content on the digital asset. And that's where the entire architectural design of the engineering comes into play, and we work very closely with our technology partners, where we are in technical terms really scoring this model in real-time, where the data is coming in in real-time, and we're able to figure out from all of the content that is available on a website or a mobile app what is that one content that we want to show up for this customer. So you would imagine that there is a full-blown capability that is sitting behind in the ecosystem which is really running all of these logics in play, and surfacing up that content for that customer. I think that's really what brings it all together.

As you would imagine, there are teams which would be data scientists, there would be teams which are marketing teams, which are really figuring out the content. There are our technology partners who are really designing this ecosystem, so that it all in all works in tandem to bring up the content on our digital asset.

**Daniel Whitenack:** Well, as Chris knows, I usually am the one that gets hung up on practicalities... And that number that you mentioned is still sticking in my brain, and this sort of real-time thing... And I know that there's a lot of people out there that are not doing that scale of inferencing and real-time sort of stuff, but they're still wanting to -- maybe they're integrating a model in their web application and they still want their web application to be responsive... Or they're trying to scale that up as their company is growing.

I wanna not miss the opportunity to ask you if, you know, as you sort of scaled up these models over time, are there any kind of practical tips that you can give practitioners, or even just team leads, any practical tips that you could give them such that they're not building models and integrations of models that they never see the benefit out of, because they take 15 seconds to do an inference, or something... You know, and they just never integrate it.

**Chris Benson:** You've done something that not many of us have done, that level of scale and performance.

**Daniel Whitenack:** Yeah, so any tips there, or anything that comes to mind in terms of guidance for teams on that subject?

**Madhurima Khandelwal:** Absolutely. I think the prerequisite of an AI/ML model is not just if you're running a real-time application. I think a lot of AI/ML models also exist even if they're running in a batch process, which means that they're running once in a month. But I think the first and foremost tip I would want to give is ask yourself whether this business problem really requires any AI/ML model or not. I think that's very critical.

\[27:48\] I think we're living in this space where sometimes AI/ML models are not very well explained, and hence, we need to be crystal clear about the data that goes into the model, the attributes that we're building basis this data, so that what comes out of the algorithm - we're very clear about what decision is it really making... And hence, once you solved the problem that "Yes, this problem requires an AI/ML model", then the question is about "Okay, what kind of techniques are out there?" Is it data which is very structured, and that's where I have labels - the 1/0, Daniel, that you were talking about - or is it unstructured, and therefore the techniques that I would like to apply is things like NLP. So once you started the technique problem, it gets into really understanding the data that was fed into it, and therefore the features that you wanna generate.

I think a lot of the practices that we used to apply for the traditional models still hold for the AI/ML models as well... But the AI/ML models give us the higher accuracy that we need. They give us an ability to work with large amounts of data, and they give us an ability to be able to churn the output in almost milliseconds, in case you have a real-time application. But it will still give you the accuracy bump, even if your application requires a batch model.

So those would be some of the tips.

**Chris Benson:** That's great information. I love the fact that you're talking about the fact that not everything needs to be an AI model, because I think that's a really core wisdom in this field, that is important not to lose sight of. Because there's a cost to deploying a deep learning model that is higher than other things... So the fact that you're recognizing that data science is larger than just this niche here.

So how do you do that when you have so many use cases to address? And you talked about starting with the fraud detection, but there are many areas that you're using various types of modeling... How do you approach an evaluation? And I don't necessarily mean just the technical evaluation, but the business evaluation in terms of there's a problem that we need to solve, and we have an array of tools which we might apply, in terms of models that we might apply to solve those. How do you make that evaluation of what should be a particular AI architecture, or say "You know what - we don't need that. We could use a standard regression on this"? How do you go through the process, regardless of what the problem is that you're addressing? How does your team address that process?

**Madhurima Khandelwal:** I think that problem at hand, Chris, is really when you're facing that problem first time. Because once you've figured it out, you would repeatedly just enhance your current logic. But first time, I would say that any team would build the best segmentation possible, the best AI/ML model possible, and if you were to just compare the two, if your AI/ML model is really able to surpass your segmentation, you would just decide the added complexity, the added cost of really implementing the AI/ML model. But if your AI/ML model is pretty much performing at the same level as a segmentation, one would question "Do you really need any AI/ML model to solve this problem or not?"

I think that's a very fair way of looking at it as well. But once you have, for this given problem, with the current set of data that exists, with the solution at hand that you have in mind, a segmentation maybe as good. And that's how you would approach it. But time changing, data quantum changing - the same problem may require an AI/ML solution. So I think we almost need to keep reevaluating the need, given the context. The context is very important.

\[31:49\] And of course, as I said, while we have people who are proficient in this field in itself for a number of years, we also have dedicated teams who play the oversight role as well. So I as a data scientist, while I would have built a model, another team would act as an oversight and make sure that what I've built and how it is being used is actually adhering to the way we want to function as a company. So that brings in that added layer of ensuring that we are solving the business problem as it needs to be.

**Break:** \[32:31\]

**Chris Benson:** So I guess to ask the next question - you're operating in this business environment where you have to deal with regulation... Daniel, you've got the trillions number stuck in my head as well... I've been thinking about that... And I'm thinking, the world that you're operating in, in terms of regulation, another new big topic is...

**Daniel Whitenack:** Auditing...

**Chris Benson:** ...now AI ethics of data... There's so many areas to dive into. You're dealing with things such as scale. Especially with ongoing regulation and with this relatively new (over the last couple of years) topic of AI ethics, how has American Express dealt and dived into and mitigated the issues and addressed the new thinking associated with this? And you can go anywhere you wanna go with this question, but I'm really curious how your team has approached the regulation and the ethical concerns.

**Madhurima Khandelwal:** Yeah, I think since the start, American Express, we have always ensured that whatever models that we build, they are free from unlawful bias. And to meet this commitment, we are very intentional in what data we do not collect, and how we build our models as well. All of the colleagues who are involved in the development, as well as maintenance of our strategies and models, they go through very vigorous training. And these trainings would include some of the fair lending laws that exist; these become prerequisite before you even initiate any form of modeling in the company.

We also conduct extensive fair lending reviews - I was talking to you about an oversight theme that exists - and that really ensures that we remain vigilant against any kind of bias. I think those are some of the steps that we've always taken as we entered into this space from traditional modeling.

**Daniel Whitenack:** I wanna follow up on that, actually, because there's something I've been thinking about probably for the last couple of years... Because there was an article by DJ Patil, who is the chief data scientist in the U.S, Hillary Mason and O'Reilly... And they talked about this idea that doing "good" data science, or ethical data science also helps you do good data science in the sense of being more proficient at the things that you're doing.

For example, Mads, you talked about knowing what model produced what result, and making sure things were tracked, and knowing what data was used, and what model... I was wondering if this is something you've seen played out on your teams, where if you do put in the effort to make sure that you're tracking your experiments, you have a really good understanding of what data is coming into and out of models, and you actually monitor those models over time and put that infrastructure in place - if that helps you when you're doing upgrades to the models, or it helps you in understanding where the models are failing such that actually in the end, if those maybe things that some people might see as burdensome, if those could actually help you in the end to do better AI, or do better data science. Is that something you've experienced?

**Madhurima Khandelwal:** \[36:09\] Yeah, all the time, in fact. I would say that often we have to understand why a card member may be approved or declined, as a simple example. What could be some of the features resulting into this decision. And the only way I'll be able to answer that question is if I would understand what's feeding into the models... And remember, these would be complex models, so I would need to then understand what exactly resulted in this decision happening.

Going back to what I was saying - if we are very clear about the attributes fitting into my model, very clear about \[unintelligible 00:36:49.03\] that instance that I'm talking about, of approval versus decline, what really contributed to that decision... That, number one, for an existing model is making me better aware of what is really feeding into that decision.

But tomorrow, as data starts to drift, as anomalies start to enter, I would then be able to understand "Now it's time for me to think about alternative feeds, alternative techniques, whatever that indication is." That's also better governed when I understand what's really entering into the ecosystem, or what is really changing or causing that delta.

So yeah, as I said, Daniel, that would be pretty much all the time, and that would be how all of our data scientists function within American Express, really knowing what exists in the models, but also monitoring it all the time, so that we are making those decisions at the right time.

**Chris Benson:** Gotcha. And I have a follow-up for that. I'm wondering - there's a problem that all of us in this industry that use these tools have got to find our way through, and that is that since you're doing inference on deep learning models, and you have the features going in, you have a lot of data going in... But though work is being done obviously on explainable AI, and there's a whole kind of mini-industry that's starting to develop to address that - how does American Express address the fact that if you're using a deep learning model that is inferencing, and you don't have that deterministic capability of explaining what happened - that requires you to have policies in place to accommodate that; how have you guys approached that? It's always interesting, because every company that deploys these at scale has to have something in mind on how they're going to address it.

If you have customers, and they're saying "Well, why did I fail a particular check?" or something like that, how do you approach that if you don't have a deterministic, explainable path to do that with? What's your approach?

**Madhurima Khandelwal:** Yeah, so as I said, as a part of AI Labs we also build platforms and products which basically help our modelers build their machine learning models. One of the things which is integrated into these platforms is their ability to look at their model scores and also interpret their model scores at scale. But at the same time, I would also say that American Express is in the process of enhancing our own internal ethical AI principles, so that we ensure colleagues across the company uphold and adhere to these values when we use AI. This is being done through a cross-functional partnership between executive leadership across our data-related organizations, as well as risk and compliance.

**Daniel Whitenack:** \[39:51\] So I'm curious, but while we still have some time to do this, but I wanna definitely give you a chance to brag a bit on your team... Because I'm looking through your website, again, on the AI Labs, and there's a section about published research and all of that, and there's just some really amazing things that seem like are going on... Like detecting sarcasm, and numerical portions of text, a tool for end-to-end distributed deep learning, there's a tool to assess availability of container-based systems, joint distributed representation of text and structure of semi-structured documents... Just a lot of cool stuff, and just to list out a few of these things. Are there any projects or breakthroughs that you'd like to brag a little bit about in terms of your team and what they've accomplished, or what they're working on?

**Madhurima Khandelwal:** So you already talked a lot about those, Daniel, but I think while we have been lately investing in a lot of AI-based automation, where we are creating a suite which would cater to a lot of our internal colleagues when they deal with really long, complex documents, I think one space in NLP that we have been working is our ability to really talk to complex data or complex reports in very simple, natural language.

This is able to surface up the needs for our senior leaders in their ability to extract information that may be the need of the hour in a fraction of a second, where we traverse through a very complex data source. In one of the recent conferences, we also presented this as a paper, where what we really have been able to implement is, again, at scale, be whatever complex data that you produce - if you want your users to understand that data and be able to extract information from it, how could you use our product or platform to be able to do that?

I can't reveal a lot of the details sitting behind, or the brain part of it, but just imagine your ability to really amplify the usage of a complex data just because you made it available to non-technical users... But even within technical users, when new members are getting on-boarded and they are getting trained on really how to work with very complex data, this product actually enables and helps them as well, and visualizes for them what they would have returned in their patch code - did that result in a similar output as this report would do.

**Daniel Whitenack:** Yeah, that's very interesting. I know that there's people working on a variety of things related to that, like generating natural language reports out of data, so like data-to-text sort of tasks... It sounds like part of what you're after is -- it's almost like you could ask questions or do some comprehensibility of complex data... But I'm sure that that data that you're searching over, I imagine it involves PDF documents, and Word documents, and videos... I'm assuming there's a whole variety of that that is internal to whatever it's called; American Express' Archive.

\[43:45\] I know we're a big enough organization where I work where we have literally what's called The Archive. And there's so much in there, but it is sometimes a chore to find that. Now, we have really amazing archive managers who pretty much can find anything for me, and they're doing that intelligence for me a lot of times... But it sounds like you're wanting to sort of enable people, give them that sort of archive specialist superpower almost. Is that right?

**Madhurima Khandelwal:** Yeah. And to add to that, if we were to work across different documents, not only in terms of types, but even in terms of languages, because we support all of the global markets, that just increases the complexity. What may work for English will definitely most likely not work for Spanish. So yeah, it definitely requires that investment of time. It also requires that expertise to really get into and extract that value for the business outcome.

**Chris Benson:** So you're already starting to address my next question, at least in the shorter-term, and that is kind of winding us up with kind of where you see things going. And you can kind of take that question any way you wanna go, in terms of where AI/how AI affects American Express, the future of business in that... I'm just really curious to see if you look out beyond things that are being productized today, and kind of the future of what some of those aspirations are that you haven't yet addressed, things that would be like "If we could do that, that would be really cool", that kind of thing... What are you thinking? What would you like to see in that medium to long-term horizon in terms of how AI impacts American Express?

**Madhurima Khandelwal:** \[unintelligible 00:45:29.23\] is that AI/ML is already quite deeply integrated in most of the functions within the finance vertical. I think I expect it to only expand even further in the future, from the core functions such as credit decisioning, we talked about fraud detection... I know, Daniel, you have that number in your mind... Marketing, servicing, even governance and compliance, we talked about those elements. But I think it's also expanding to some of the ancillary functions such as process automation, cloud strategy... And I think AI/ML is truly modernizing and streamlining finance as we think about it.

For American Express, I would say it's really these three themes that matter to us - we want to use our data assets with the freshest data possible to make it real-time decisions. Second, we wanna produce data products at scale, so that we are always improving the quality, and third, we wanna double down on improving customer service and experiences. I think these are really at the heart of it, and I can just imagine a future where we will keep investing in the space of AI/ML.

**Daniel Whitenack:** Awesome. That's really great to hear, and I know I do, and I'm sure Chris does appreciate your perspective on these topics; being in a position where you have scaled some of these things up, it's just really tremendous to get that perspective and understand some of those things. Thank you for taking time near holidays and winter break to chat with us about these things; it's been really great, and we hope you get some time off before the new year starts. Thank you so much for the insights.

**Madhurima Khandelwal:** Thank you so much for having me. This was amazing.
