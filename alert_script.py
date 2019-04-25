class Alert:

    sender_email = 'habtest3@gmail.com'
    sender_phone = '+19093219750'
    account_sid = 'ACfe3f81c3a2cd6eccfd1ac3a8f2449584'
    auth_token = '8ba90a5067ff1ec9e5ecc22397e6c999'

    def send_email(self,receivers = None, \
        subject_txt = None, msg = None, file_path = None):

        import yagmail
        yag = yagmail.SMTP(Alert.sender_email)
        yag.send(
            to=receivers,
            subject=subject_txt,
            contents=msg,
            attachments=file_path 
        )
    
    def send_text(self, msg, receivers):

        from twilio.rest import Client
        client = Client(Alert.account_sid, Alert.auth_token)

        for each in receivers:
            client.messages \
                .create(
                    body=msg,
                    from_=Alert.sender_phone,
                    to=each
                )

