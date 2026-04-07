import re
from urllib.parse import urlsplit

from utils.text_model_service import TextModelService


class TextRiskAnalyzer:
    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    URL_PATTERN = re.compile(r"(https?://[^\s]+|www\.[^\s]+|\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?\b)")
    EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    PHONE_CANDIDATE_PATTERN = re.compile(r"(?:\+\d[\d\s-]{3,20}\d|\b\d[\d\s-]{3,20}\d\b|\b\d{3,4}\b)")
    MONEY_PATTERN = re.compile(r"(\$\s?\d+(?:[\.,]\d{1,2})?|\b\d+(?:[\.,]\d{1,2})?\s?(?:usd|sgd|rm)\b)", re.IGNORECASE)
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?:[!?;]|\.(?=\s)|\n)+")

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = TextModelService(
            cfg.TEXT_PHISHING_MODEL_NAME,
            cfg.DEVICE,
            cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX,
        )
        self.chinese_model = TextModelService(
            getattr(cfg, "CHINESE_TEXT_PHISHING_MODEL_NAME", ""),
            cfg.DEVICE,
            getattr(cfg, "CHINESE_TEXT_MODEL_POSITIVE_CLASS_INDEX", None),
        ) if getattr(cfg, "CHINESE_TEXT_PHISHING_MODEL_NAME", "") else None
        self.trusted_email_addresses = {
            self._normalize_email(email)
            for email in getattr(cfg, "TRUSTED_EMAIL_ADDRESSES", ())
            if self._normalize_email(email)
        }
        self.trusted_email_domains = {
            self._normalize_domain(domain)
            for domain in getattr(cfg, "TRUSTED_EMAIL_DOMAINS", ())
            if self._normalize_domain(domain)
        }
        self.trusted_phone_numbers = {
            self._normalize_phone_number(phone)
            for phone in getattr(cfg, "TRUSTED_PHONE_NUMBERS", ())
            if self._normalize_phone_number(phone)
        }
        self.trusted_url_domains = {
            self._normalize_domain(domain)
            for domain in getattr(cfg, "TRUSTED_URL_DOMAINS", ())
            if self._normalize_domain(domain)
        }
        self.all_keyword_terms = tuple(dict.fromkeys(
            term
            for terms in getattr(cfg, "PHISHING_KEYWORDS", {}).values()
            for term in terms
        ))
        self.url_tld_allowlist = tuple(sorted(
            getattr(cfg, "URL_TLD_ALLOWLIST", ()),
            key=len,
            reverse=True,
        ))

    def analyze(self, text: str):
        normalized = (text or "").strip()
        relevant_chunks, dropped_chunks = self._select_relevant_chunks(normalized)
        filtered_text = " ".join(relevant_chunks).strip()
        lowered = filtered_text.lower()
        denoised = self._denoise_for_keywords(lowered)

        matched_groups = {}
        total_hits = 0

        for group_name, keywords in self.cfg.PHISHING_KEYWORDS.items():
            hits = [
                keyword for keyword in keywords
                if self._contains_term(lowered, keyword) or self._contains_term(denoised, keyword)
            ]
            if hits:
                matched_groups[group_name] = hits
                total_hits += len(hits)

        detected_urls = self._extract_urls(filtered_text)
        detected_emails = self.EMAIL_PATTERN.findall(filtered_text)
        detected_phone_numbers = self._extract_phone_numbers(filtered_text)
        trusted_urls, urls = self._partition_trusted_urls(detected_urls)
        trusted_emails, emails = self._partition_trusted_emails(detected_emails)
        trusted_phone_numbers, phone_numbers = self._partition_trusted_phone_numbers(detected_phone_numbers)
        money_mentions = self.MONEY_PATTERN.findall(filtered_text)
        benign_hits = self._collect_term_hits(
            lowered,
            getattr(self.cfg, "BENIGN_CONTEXT_SIGNALS", ()),
            alternate_text=denoised,
        )
        benign_blockers = self._collect_term_hits(
            lowered,
            getattr(self.cfg, "BENIGN_PENALTY_BLOCKERS", ()),
            alternate_text=denoised,
        )
        trusted_signal_count = sum([
            bool(trusted_urls),
            bool(trusted_emails),
            bool(trusted_phone_numbers),
        ])
        benign_penalty = 0.0
        if benign_hits and not benign_blockers:
            benign_penalty += len(benign_hits) * self.cfg.BENIGN_CONTEXT_HIT_WEIGHT
        if not benign_blockers:
            trusted_bonus_count = sum([
                bool(trusted_urls),
                bool(trusted_emails),
                bool(trusted_phone_numbers) and bool(benign_hits),
            ])
            if trusted_bonus_count:
                benign_penalty += trusted_bonus_count * self.cfg.TRUSTED_SIGNAL_BENIGN_BONUS
        benign_penalty = min(benign_penalty, self.cfg.BENIGN_CONTEXT_MAX_PENALTY)
        phone_signal_active = self._should_treat_phone_as_risky(
            filtered_text,
            phone_numbers,
            matched_groups,
            urls,
            emails,
            money_mentions,
        )

        rule_score = 0.0
        rule_score += min(total_hits * self.cfg.KEYWORD_HIT_WEIGHT, 0.6)

        if urls:
            rule_score += self.cfg.URL_PRESENT_WEIGHT
        if emails:
            rule_score += self.cfg.EMAIL_PRESENT_WEIGHT
        if phone_signal_active:
            rule_score += self.cfg.PHONE_PRESENT_WEIGHT
        if money_mentions:
            rule_score += self.cfg.MONEY_PRESENT_WEIGHT

        signal_families = sum([
            bool(matched_groups),
            bool(urls),
            bool(emails),
            bool(phone_signal_active),
            bool(money_mentions),
        ])
        if signal_families >= 3:
            rule_score += self.cfg.MULTI_SIGNAL_BONUS
        if ("credential_request" in matched_groups) and (urls or emails):
            rule_score += self.cfg.LINK_CREDENTIAL_BONUS
        if benign_penalty:
            rule_score = max(rule_score - benign_penalty, 0.0)

        if not filtered_text:
            rule_score *= self.cfg.EMPTY_TEXT_PENALTY

        rule_score = min(rule_score, 1.0)

        model_input_text = filtered_text
        masked_trusted_contacts = []
        if getattr(self.cfg, "MASK_TRUSTED_CONTACTS_FOR_MODEL", False):
            model_input_text, masked_trusted_contacts = self._mask_trusted_contacts_for_model(
                filtered_text,
                trusted_urls,
                trusted_emails,
                trusted_phone_numbers,
            )
        masked_phone_numbers_for_model = []
        if getattr(self.cfg, "MASK_ALL_PHONE_NUMBERS_FOR_MODEL", False):
            model_input_text, masked_phone_numbers_for_model = self._mask_phone_numbers_for_model(
                model_input_text,
                detected_phone_numbers,
            )

        model_chunks = self._split_text_for_model(model_input_text)
        chunk_scores = []
        for chunk in model_chunks:
            prob, route, scored_text = self._predict_chunk_probability(chunk)
            if prob is not None:
                chunk_scores.append((chunk, prob, route, scored_text))

        if chunk_scores:
            best_chunk, model_score_raw, model_route, model_scored_text = max(chunk_scores, key=lambda x: x[1])
        else:
            best_chunk, model_score_raw, model_route, model_scored_text = (None, None, None, None)

        model_score = self._calibrate_model_score(model_score_raw)
        if model_score is not None and benign_penalty:
            model_score = max(
                model_score - benign_penalty * self.cfg.BENIGN_MODEL_PENALTY_MULTIPLIER,
                0.0,
            )

        if model_score is None:
            score = rule_score
        else:
            # Model signal is always included; rules are not used as a hard gate.
            score = (
                self.cfg.TEXT_RULE_WEIGHT * rule_score
                + self.cfg.TEXT_MODEL_WEIGHT * model_score
            )
        score = min(score, 1.0)

        reasons = []
        if matched_groups:
            reasons.append("phishing language detected")
        if urls:
            reasons.append("URL detected in image text")
        if emails:
            reasons.append("email address detected in image text")
        if phone_signal_active:
            reasons.append("phone number detected in image text")
        elif phone_numbers:
            reasons.append("phone number detected but context looks benign")
        if trusted_urls:
            reasons.append("trusted government site ignored for heuristic scoring")
        if trusted_emails:
            reasons.append("trusted email/domain ignored for heuristic scoring")
        if trusted_phone_numbers:
            reasons.append("trusted phone number ignored for heuristic scoring")
        if benign_hits:
            reasons.append("benign notification context detected")
        if masked_trusted_contacts:
            reasons.append("trusted contacts masked before phishing-model scoring")
        if masked_phone_numbers_for_model:
            reasons.append("all detected phone numbers masked before phishing-model scoring")
        if money_mentions:
            reasons.append("money-related language detected")
        if signal_families >= 3:
            reasons.append("multiple independent phishing signal types detected")
        if model_score_raw is not None:
            reasons.append(
                f"{self._describe_model_route(model_route)} probability: {model_score_raw:.2f} "
                f"(calibrated: {model_score:.2f})"
            )
            if best_chunk:
                reasons.append(f"model strongest text span: '{best_chunk[:90]}'")
        elif self.model.is_loaded or (self.chinese_model and self.chinese_model.is_loaded):
            reasons.append("text model loaded but no usable text for inference")
        else:
            reasons.append("text model unavailable; heuristic-only text scoring")
        if relevant_chunks:
            reasons.append(f"relevance filter kept {len(relevant_chunks)} chunk(s)")
        if dropped_chunks:
            reasons.append(f"relevance filter dropped {len(dropped_chunks)} chunk(s)")
        warnings = []
        if model_score is not None and model_score >= 0.6 and rule_score < 0.18:
            warnings.append("model indicates phishing but rule signals are weak")
        if model_score is not None and model_score < 0.2 and rule_score >= 0.4:
            warnings.append("rule signals are strong but model confidence is low")
        if warnings:
            reasons.extend([f"warning: {w}" for w in warnings])
        if not reasons:
            reasons.append("no obvious phishing text signals found")

        return {
            "score": round(score, 4),
            "rule_score": round(rule_score, 4),
            "model_score": None if model_score is None else round(model_score, 4),
            "model_score_raw": None if model_score_raw is None else round(model_score_raw, 4),
            "model_loaded": self.model.is_loaded or bool(self.chinese_model and self.chinese_model.is_loaded),
            "english_model_loaded": self.model.is_loaded,
            "chinese_model_loaded": bool(self.chinese_model and self.chinese_model.is_loaded),
            "model_route": model_route,
            "model_scored_text": model_scored_text,
            "model_chunks_evaluated": len(chunk_scores),
            "model_best_chunk": best_chunk,
            "input_text": normalized,
            "filtered_text": filtered_text,
            "model_input_text": model_input_text,
            "relevant_chunks": relevant_chunks,
            "dropped_chunks": dropped_chunks,
            "warnings": warnings,
            "keyword_groups": matched_groups,
            "urls": urls,
            "trusted_urls": trusted_urls,
            "emails": emails,
            "phone_numbers": phone_numbers if phone_signal_active else [],
            "detected_phone_numbers": detected_phone_numbers,
            "context_benign_phone_numbers": [] if phone_signal_active else phone_numbers,
            "trusted_emails": trusted_emails,
            "trusted_phone_numbers": trusted_phone_numbers,
            "masked_trusted_contacts": masked_trusted_contacts,
            "masked_phone_numbers_for_model": masked_phone_numbers_for_model,
            "detected_urls": detected_urls,
            "detected_emails": detected_emails,
            "benign_context_hits": benign_hits,
            "benign_penalty_blockers": benign_blockers,
            "money_mentions": money_mentions,
            "reasons": reasons,
        }

    def _denoise_for_keywords(self, text: str) -> str:
        char_map = str.maketrans({
            "0": "o",
            "1": "i",
            "!": "i",
            "|": "l",
            "$": "s",
            "5": "s",
            "@": "a",
            "3": "e",
            "7": "t",
        })
        mapped = text.translate(char_map)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", mapped)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _calibrate_model_score(self, score):
        if score is None:
            return None
        threshold = self.cfg.MODEL_POSITIVE_THRESHOLD
        if score <= threshold:
            return 0.0
        return min((score - threshold) / (1.0 - threshold), 1.0)

    def _select_relevant_chunks(self, text: str):
        normalized = (text or "").strip()
        if not normalized:
            return [], []

        chunks = self._candidate_relevance_chunks(normalized)
        scored = []
        for chunk in chunks:
            score = self._score_chunk_relevance(chunk)
            scored.append((chunk, score))

        kept = [
            chunk for chunk, score in scored
            if score >= self.cfg.TEXT_RELEVANCE_MIN_SCORE
        ]
        kept = kept[:self.cfg.TEXT_RELEVANCE_MAX_CHUNKS]

        if not kept and scored:
            best_chunk, best_score = max(scored, key=lambda item: item[1])
            if best_score > 0:
                kept = [best_chunk]
            elif self.CHINESE_PATTERN.search(normalized):
                # Keep the best Chinese-bearing chunk so the Chinese classifier
                # still gets a chance even when English-centric heuristics score low.
                kept = [best_chunk]

        kept_keys = {chunk.lower() for chunk in kept}
        dropped = [chunk for chunk, _ in scored if chunk.lower() not in kept_keys]
        return kept, dropped

    def _candidate_relevance_chunks(self, text: str):
        chunks = []
        chunks.extend([
            seg.strip()
            for seg in self.SENTENCE_SPLIT_PATTERN.split(text)
            if seg.strip()
        ])

        tokens = text.split()
        window = 12
        stride = 6
        if len(tokens) > window:
            for i in range(0, len(tokens), stride):
                piece = " ".join(tokens[i:i + window]).strip()
                if piece:
                    chunks.append(piece)

        deduped = []
        seen = set()
        for chunk in chunks:
            key = chunk.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(chunk)
        return deduped

    def _score_chunk_relevance(self, chunk: str):
        lowered = chunk.lower()
        denoised = self._denoise_for_keywords(lowered)
        score = 0.0

        for term in self.cfg.TEXT_RELEVANCE_STRONG_SIGNALS:
            if self._contains_term(lowered, term) or self._contains_term(denoised, term):
                score += 2.0
        for term in self.cfg.TEXT_RELEVANCE_WEAK_SIGNALS:
            if self._contains_term(lowered, term) or self._contains_term(denoised, term):
                score += 0.75
        for term in getattr(self.cfg, "TEXT_RELEVANCE_STRONG_SIGNALS_ZH", ()):
            if term in chunk:
                score += 2.0
        keyword_hits = self._collect_term_hits(
            lowered,
            self.all_keyword_terms,
            alternate_text=denoised,
        )
        score += min(len(keyword_hits) * 1.25, 4.0)
        for term in self.cfg.TEXT_RELEVANCE_NOISE_HINTS:
            if self._contains_term(lowered, term) or self._contains_term(denoised, term):
                score -= 1.5

        _, untrusted_urls = self._partition_trusted_urls(self._extract_urls(chunk))
        _, untrusted_emails = self._partition_trusted_emails(self.EMAIL_PATTERN.findall(chunk))
        _, untrusted_phone_numbers = self._partition_trusted_phone_numbers(self._extract_phone_numbers(chunk))
        if untrusted_urls:
            score += 2.5
        if untrusted_emails:
            score += 2.0
        if self._should_treat_phone_as_risky(chunk, untrusted_phone_numbers):
            score += 0.75
        if self.MONEY_PATTERN.search(chunk):
            score += 1.5

        token_count = len(chunk.split())
        if token_count < 3:
            score -= 1.5
        elif token_count > 25:
            score -= 0.5

        alpha_count = sum(ch.isalpha() for ch in chunk)
        junk_count = len(re.findall(r"[^A-Za-z0-9\s\u4e00-\u9fff.,:;!?$%&@()+/\-_'\"]", chunk))
        if alpha_count and junk_count > alpha_count * 0.2:
            score -= 1.0

        return score

    def _contains_term(self, text: str, term: str) -> bool:
        haystack = str(text or "").lower()
        needle = str(term or "").strip().lower()
        if not haystack or not needle:
            return False
        if self.CHINESE_PATTERN.search(needle):
            return needle in haystack
        pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
        if re.search(pattern, haystack) is not None:
            return True
        if any(ch.isspace() for ch in needle) or any(ch in needle for ch in "-._/"):
            collapsed_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
            collapsed_needle = re.sub(r"[^a-z0-9]+", "", needle)
            if collapsed_needle and collapsed_needle in collapsed_haystack:
                return True
        return False

    def _collect_term_hits(self, text: str, terms, alternate_text: str | None = None) -> list[str]:
        hits = []
        for term in terms:
            if self._contains_term(text, term) or (alternate_text and self._contains_term(alternate_text, term)):
                hits.append(term)
        return hits

    def _should_treat_phone_as_risky(
        self,
        text: str,
        phone_numbers,
        matched_groups=None,
        urls=None,
        emails=None,
        money_mentions=None,
    ) -> bool:
        if not phone_numbers:
            return False

        lowered = str(text or "").lower()
        denoised = self._denoise_for_keywords(lowered)
        benign_hits = self._collect_term_hits(
            lowered,
            getattr(self.cfg, "PHONE_BENIGN_CONTEXT_SIGNALS", ()),
            alternate_text=denoised,
        )
        risky_hits = self._collect_term_hits(
            lowered,
            getattr(self.cfg, "PHONE_RISK_CONTEXT_SIGNALS", ()),
            alternate_text=denoised,
        )

        if benign_hits and not risky_hits:
            return False

        matched_groups = matched_groups or {}
        urls = urls or []
        emails = emails or []
        money_mentions = money_mentions or []
        suspicious_group_names = {
            "urgency",
            "credential_request",
            "financial_request",
            "call_to_action",
            "prize_bait",
        }
        group_support = any(group in matched_groups for group in suspicious_group_names)
        transport_support = bool(urls or emails)
        monetary_support = bool(money_mentions) and not benign_hits

        if benign_hits and not (transport_support or group_support):
            return False

        return bool(risky_hits or transport_support or group_support or monetary_support)

    def _extract_urls(self, text: str):
        found = []
        seen = set()

        source = text or ""
        for match in self.URL_PATTERN.finditer(source):
            candidate = str(match.group(0)).strip().strip(".,;:!?()[]{}<>\"'")
            if not candidate or "@" in candidate:
                continue
            if match.start() > 0 and source[match.start() - 1] == "@":
                continue

            domain = self._normalize_url_domain(candidate)
            if not domain:
                continue
            tld = domain.rsplit(".", 1)[-1]
            if tld not in getattr(self.cfg, "URL_TLD_ALLOWLIST", ()):
                continue

            key = candidate.lower()
            if key in seen:
                continue

            seen.add(key)
            found.append(candidate)

        return found

    def _extract_phone_numbers(self, text: str):
        found = []
        seen = set()

        for match in self.PHONE_CANDIDATE_PATTERN.findall(text or ""):
            candidate = str(match).strip().strip(".,;:!?()[]{}<>\"'")
            if not candidate:
                continue
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate) or re.fullmatch(r"\d{1,2}-\d{1,2}-\d{2,4}", candidate):
                continue

            normalized = self._normalize_phone_number(candidate)
            if not normalized:
                continue

            # Keep typical phone numbers, plus short trusted hotlines like 1799.
            if len(normalized) < 8 and normalized not in self.trusted_phone_numbers:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            found.append(candidate)

        return found

    def _partition_trusted_urls(self, urls):
        trusted = []
        untrusted = []
        for url in urls:
            domain = self._normalize_url_domain(url)
            is_trusted = any(
                domain == trusted_domain or domain.endswith(f".{trusted_domain}")
                for trusted_domain in self.trusted_url_domains
            )
            if is_trusted:
                trusted.append(url)
            else:
                untrusted.append(url)
        return trusted, untrusted

    def _predict_chunk_probability(self, chunk: str):
        normalized = (chunk or "").strip()
        if not normalized:
            return None, None, None

        contains_chinese = bool(self.CHINESE_PATTERN.search(normalized))
        should_route_chinese = self._should_route_to_chinese_model(normalized)
        if should_route_chinese and self.chinese_model and self.chinese_model.is_loaded:
            prob = self.chinese_model.predict_phishing_probability(normalized)
            return prob, "chinese", normalized

        english_text = normalized
        route = "english"
        if contains_chinese:
            english_text = self._strip_chinese_for_english_model(normalized)
            route = "english_stripped_fallback"

        if not english_text:
            return None, None, None

        prob = self.model.predict_phishing_probability(english_text)
        return prob, route, english_text

    def _partition_trusted_emails(self, emails):
        trusted = []
        untrusted = []
        for email in emails:
            normalized = self._normalize_email(email)
            domain = normalized.split("@", 1)[1] if "@" in normalized else ""
            is_trusted = (
                normalized in self.trusted_email_addresses
                or any(
                    domain == trusted_domain or domain.endswith(f".{trusted_domain}")
                    for trusted_domain in self.trusted_email_domains
                )
            )
            if is_trusted:
                trusted.append(email)
            else:
                untrusted.append(email)
        return trusted, untrusted

    def _partition_trusted_phone_numbers(self, phone_numbers):
        trusted = []
        untrusted = []
        for phone_number in phone_numbers:
            normalized = self._normalize_phone_number(phone_number)
            if normalized and normalized in self.trusted_phone_numbers:
                trusted.append(phone_number)
            else:
                untrusted.append(phone_number)
        return trusted, untrusted

    def _normalize_email(self, email: str) -> str:
        return str(email or "").strip().lower()

    def _normalize_domain(self, domain: str) -> str:
        normalized = str(domain or "").strip().lower()
        return normalized.lstrip("@")

    def _normalize_phone_number(self, phone_number: str) -> str:
        return re.sub(r"\D+", "", str(phone_number or ""))

    def _strip_chinese_for_english_model(self, text: str) -> str:
        stripped = self.CHINESE_PATTERN.sub(" ", str(text or ""))
        return re.sub(r"\s+", " ", stripped).strip()

    def _normalize_url_domain(self, url: str) -> str:
        candidate = str(url or "").strip().lower().strip(".,;:!?()[]{}<>\"'")
        if not candidate:
            return ""

        with_scheme = candidate if "://" in candidate else f"https://{candidate}"

        try:
            parsed = urlsplit(with_scheme)
        except Exception:
            return ""

        domain = parsed.netloc or parsed.path.split("/", 1)[0]
        domain = domain.split("@")[-1].split(":", 1)[0].strip(".")
        return self._repair_ocr_merged_tld(domain)

    def _repair_ocr_merged_tld(self, domain: str) -> str:
        labels = [label for label in str(domain or "").split(".") if label]
        if not labels:
            return ""

        last_label = labels[-1]
        if last_label in self.url_tld_allowlist:
            return ".".join(labels)

        for tld in self.url_tld_allowlist:
            if last_label.startswith(tld) and len(last_label) > len(tld):
                labels[-1] = tld
                return ".".join(labels)

        return ".".join(labels)

    def _should_route_to_chinese_model(self, text: str) -> bool:
        normalized = str(text or "")
        chinese_chars = len(self.CHINESE_PATTERN.findall(normalized))
        if chinese_chars == 0:
            return False

        alpha_or_chinese = [
            ch for ch in normalized
            if ch.isalpha() or self.CHINESE_PATTERN.match(ch)
        ]
        total_letters = len(alpha_or_chinese)
        chinese_ratio = chinese_chars / total_letters if total_letters else 1.0

        min_chars = getattr(self.cfg, "CHINESE_ROUTE_MIN_CHAR_COUNT", 1)
        min_ratio = getattr(self.cfg, "CHINESE_ROUTE_MIN_CHAR_RATIO", 0.0)
        return chinese_chars >= min_chars and chinese_ratio >= min_ratio

    def _mask_trusted_contacts_for_model(self, text: str, trusted_urls, trusted_emails, trusted_phone_numbers):
        masked = str(text or "")
        replaced = []

        for url in sorted(set(trusted_urls), key=len, reverse=True):
            if url and url in masked:
                masked = masked.replace(url, "[TRUSTED_URL]")
                replaced.append(url)

        for email in sorted(set(trusted_emails), key=len, reverse=True):
            if email and email in masked:
                masked = masked.replace(email, "[TRUSTED_EMAIL]")
                replaced.append(email)

        for phone_number in sorted(set(trusted_phone_numbers), key=len, reverse=True):
            if phone_number and phone_number in masked:
                masked = masked.replace(phone_number, "[TRUSTED_PHONE]")
                replaced.append(phone_number)

        return masked, replaced

    def _mask_phone_numbers_for_model(self, text: str, phone_numbers):
        masked = str(text or "")
        replaced = []

        for phone_number in sorted(set(phone_numbers or ()), key=len, reverse=True):
            if phone_number and phone_number in masked:
                masked = masked.replace(phone_number, "[PHONE_NUMBER]")
                replaced.append(phone_number)

        return masked, replaced

    def _describe_model_route(self, route: str | None) -> str:
        labels = {
            "english": "English DistilBERT model",
            "english_stripped_fallback": "English DistilBERT fallback",
            "chinese": "Chinese text model",
        }
        return labels.get(route or "", "Text model")

    def _split_text_for_model(self, text: str):
        normalized = (text or "").strip()
        if not normalized:
            return []

        chunks = [normalized]

        # Split into sentence-like segments first.
        segments = [
            seg.strip()
            for seg in self.SENTENCE_SPLIT_PATTERN.split(normalized)
            if seg.strip()
        ]
        chunks.extend(segments)

        # Add token windows so short phishing spans inside long OCR strings are not diluted.
        tokens = normalized.split()
        window = 16
        stride = 8
        if len(tokens) > window:
            for i in range(0, len(tokens), stride):
                piece = " ".join(tokens[i:i + window]).strip()
                if piece:
                    chunks.append(piece)

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for c in chunks:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped
