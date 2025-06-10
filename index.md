---
layout: default
title: About Artificial Intelligence
description: Different Kinds of Artifical Intelligence
---

# About Artificial Intelligence

This Github website currently contains two part.

{% for doc in site.ToNN %}
  {% if doc.title == "ToNN" %}
    <h2><a href="{{ doc.url | relative_url }}">Neural Network Model Study</a></h2>
  {% endif %}
{% endfor %}

Have studied different kinds of Neural Network Models

<ul>
  {% for doc in site.ToNN %}
    <li>
      <a href="{{ doc.url | relative_url }}">{{ doc.title }}</a>
    </li>
  {% endfor %}
</ul>

{% for doc in site.FAS %}
  {% if doc.title == "FAS" %}
    <h2><a href="{{ doc.url | relative_url }}">Face Anti-Spoofing</a></h2>
  {% endif %}
{% endfor %}

<ul>
  {% for doc in site.FAS %}
    <li>
      <a href="{{ doc.url | relative_url }}">{{ doc.title }}</a>
    </li>
  {% endfor %}
</ul>
